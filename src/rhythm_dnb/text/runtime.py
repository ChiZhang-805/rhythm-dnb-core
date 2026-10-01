"""Single-device and torchrun execution, with exact sample accounting across ranks."""

from contextlib import nullcontext
from datetime import timedelta
import os
import sys
import torch
import torch.distributed as distributed


def hardware_report():
    # PSEUDOCODE: inspect devices visible on this host without starting training or reserving GPU memory.
    available = torch.cuda.is_available()
    devices = []
    for index in range(torch.cuda.device_count() if available else 0):
        properties = torch.cuda.get_device_properties(index)
        devices.append({'index': index, 'name': properties.name, 'memory_bytes': properties.total_memory,
                        'compute_capability': list(torch.cuda.get_device_capability(index))})
    return {'platform': sys.platform, 'torch': torch.__version__, 'cuda_build': torch.version.cuda,
            'cuda_available': available, 'visible_devices': devices,
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'multi_gpu_available': available and len(devices) > 1 and sys.platform != 'win32' and distributed.is_nccl_available()}


class TrainingRuntime:
    def __init__(self, config):
        # PSEUDOCODE: record requested execution options without opening a process group.
        self.config = config
        self.owns_group = False
        self.rank = 0
        self.world_size = 1

    def __enter__(self):
        # PSEUDOCODE: bind one process to one device -> initialize communication -> agree on precision.
        requested = self.config['device']
        use_cuda = requested != 'cpu' and torch.cuda.is_available()
        if requested == 'cuda' and not use_cuda:
            raise ValueError('CUDA was requested but is unavailable.')
        self.world_size = int(os.environ.get('WORLD_SIZE', '1'))
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        if self.world_size < 1 or local_rank < 0:
            raise ValueError('Invalid distributed process environment.')
        if use_cuda:
            if local_rank >= torch.cuda.device_count():
                raise ValueError('Each process requires one visible GPU; check torchrun and CUDA_VISIBLE_DEVICES.')
            torch.cuda.set_device(local_rank)
        self.device = torch.device('cuda', local_rank) if use_cuda else torch.device('cpu')
        if self.world_size > 1 and not distributed.is_initialized():
            if sys.platform == 'win32':
                raise ValueError('Distributed training requires Linux/WSL; Windows supports single-device training.')
            if any(name not in os.environ for name in ('RANK', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT')):
                raise ValueError('Launch multiple processes with torchrun.')
            if use_cuda and not distributed.is_nccl_available():
                raise ValueError('Multi-GPU training requires Linux/WSL with the PyTorch NCCL backend.')
            distributed.init_process_group('nccl' if use_cuda else 'gloo', timeout=timedelta(minutes=self.config.get('process_timeout_minutes', 120)))
            self.owns_group = True
        if distributed.is_initialized():
            self.rank = distributed.get_rank()
            self.world_size = distributed.get_world_size()
        capable = use_cuda and torch.cuda.is_bf16_supported()
        capabilities = self.gather(capable)
        precision = self.config['precision']
        if precision == 'auto':
            precision = ('bf16' if all(capabilities) else 'fp16') if use_cuda else 'fp32'
        if precision != 'fp32' and not use_cuda:
            self.__exit__(None, None, None)
            raise ValueError('Mixed precision requires a CUDA device in this trainer.')
        if precision == 'bf16' and not all(capabilities):
            self.__exit__(None, None, None)
            raise ValueError('Requested bf16 is not supported on every participating GPU.')
        self.precision = precision
        self.dtype = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}[precision]
        return self

    def __exit__(self, exc_type, exc, traceback):
        # PSEUDOCODE: release only the communication group created by this training invocation.
        if self.owns_group and distributed.is_initialized():
            distributed.destroy_process_group()

    def gather(self, value):
        # PSEUDOCODE: gather small trusted process metadata in rank order.
        if self.world_size == 1:
            return [value]
        values = [None] * self.world_size
        distributed.all_gather_object(values, value)
        return values

    def primary(self, operation):
        # PSEUDOCODE: execute filesystem/evaluation work once -> share success or failure with every rank.
        payload = [None]
        if self.rank == 0:
            try:
                payload[0] = (True, operation())
            except Exception as error:
                payload[0] = (False, type(error).__name__ + ': ' + str(error))
        if self.world_size > 1:
            distributed.broadcast_object_list(payload, src=0)
        success, result = payload[0]
        if not success:
            raise RuntimeError('Primary training process failed: ' + result)
        return result

    def sum(self, values):
        # PSEUDOCODE: combine sample counts and loss totals without averaging process means.
        tensor = torch.tensor(values, dtype=torch.float64, device=self.device)
        if self.world_size > 1:
            distributed.all_reduce(tensor)
        return tensor.cpu().tolist()

    def autocast(self):
        # PSEUDOCODE: enable the agreed reduced precision only on CUDA devices.
        return torch.autocast('cuda', dtype=self.dtype) if self.precision != 'fp32' else nullcontext()

    def describe(self):
        # PSEUDOCODE: preserve actual device identities, precision and global batch size for reproducibility.
        from importlib.metadata import version, PackageNotFoundError
        libraries = {}
        for name in ('transformers', 'peft', 'bitsandbytes', 'safetensors', 'numpy'):
            try:
                libraries[name] = version(name)
            except PackageNotFoundError:
                libraries[name] = None
        device = {'rank': self.rank, 'device': str(self.device), 'name': 'CPU', 'memory_bytes': None}
        if self.device.type == 'cuda':
            properties = torch.cuda.get_device_properties(self.device)
            device.update(name=properties.name, memory_bytes=properties.total_memory)
        return {'devices': self.gather(device), 'world_size': self.world_size, 'precision': self.precision,
                'python': sys.version.split()[0], 'torch': torch.__version__, 'cuda_build': torch.version.cuda, 'libraries': libraries,
                'batch_size_per_device': self.config['batch_size'], 'gradient_accumulation': self.config['gradient_accumulation'],
                'effective_batch_size': self.config['batch_size'] * self.config['gradient_accumulation'] * self.world_size,
                'gradient_checkpointing': self.config['gradient_checkpointing']}


class ShardedBatches:
    """Visit each real sample once; an empty last shard gets a zero-weight synchronization item."""

    def __init__(self, size, batch_size, rank, world_size, seed, epoch):
        # PSEUDOCODE: retain one common epoch permutation and disjoint per-rank slices.
        if size < 1 or batch_size < 1 or not 0 <= rank < world_size:
            raise ValueError('Invalid training batch layout.')
        self.size, self.batch_size, self.rank, self.world_size = size, batch_size, rank, world_size
        self.seed, self.epoch = seed, epoch

    def __len__(self):
        # PSEUDOCODE: count global microbatches, including the final incomplete batch.
        width = self.batch_size * self.world_size
        return (self.size + width - 1) // width

    def __iter__(self):
        # PSEUDOCODE: shuffle identically -> shard without duplication -> zero-weight only empty ranks.
        order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + self.epoch)).tolist()
        width = self.batch_size * self.world_size
        for start in range(0, self.size, width):
            shard = order[start + self.rank * self.batch_size:min(start + (self.rank + 1) * self.batch_size, self.size)]
            yield [(index, 1.) for index in shard] if shard else [(order[0], 0.)]

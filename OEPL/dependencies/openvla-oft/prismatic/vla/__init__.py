# Avoid importing dataset tooling (and TF/dlimp deps) at module import time.
# from .materialize import get_vla_dataset_and_collator

# Lazy accessor to keep inference-only paths TF-free.
def get_vla_dataset_and_collator(*args, **kwargs):
    from .materialize import get_vla_dataset_and_collator as _impl
    return _impl(*args, **kwargs)

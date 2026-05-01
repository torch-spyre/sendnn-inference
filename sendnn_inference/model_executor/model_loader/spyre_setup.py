import os


def spyre_setup():
    # default to senulator backend unless user specified otherwise
    os.environ.setdefault("FLEX_COMPUTE", "SENULATOR")

def spyre_dist_setup(rank=0, world_size=1, local_rank=0, local_size=1, verbose=False):
    # make sure to have torchrun env vars for flex
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world_size))

    if verbose:
        print(f"Distributed rank {os.environ['RANK']} / {os.environ['WORLD_SIZE']}")

    spyre_setup()

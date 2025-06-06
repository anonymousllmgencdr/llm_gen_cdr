import os, sys
from src.train.tuner import run_exp
from src.extras.misc import get_device_count
from src.extras.logging import get_logger
import subprocess
import random
import launch


logger = get_logger(__name__)


force_torchrun = os.environ.get("FORCE_TORCHRUN", "0").lower() in ["true", "1"]
if force_torchrun or get_device_count() > 1:
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", str(random.randint(20001, 29999)))
    logger.info("Initializing distributed tasks at: {}:{}".format(master_addr, master_port))
    # print("Initializing distributed tasks at: {}:{}".format(master_addr, master_port))
    process = subprocess.run(
        (
            "torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
            "--master_addr {master_addr} --master_port {master_port} {file_name} {args}"
        ).format(
            nnodes=os.environ.get("NNODES", "1"),
            node_rank=os.environ.get("RANK", "0"),
            nproc_per_node=os.environ.get("NPROC_PER_NODE", str(get_device_count())),
            master_addr=master_addr,
            master_port=master_port,
            file_name=launch.__file__,
            args=" ".join(sys.argv[1:]),
        ),
        shell=True,
    )
    sys.exit(process.returncode)
else:
    run_exp()

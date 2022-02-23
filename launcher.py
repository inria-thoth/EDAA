"""
Script to launch jobs on THOTH cluster using Hydra
"""

import logging
import os
import shutil
import subprocess

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig
from omegaconf.errors import MissingMandatoryValue

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# TODO
# THOTH GPU
def filter_fn(s):
    return not (s.startswith("launcher") or s.startswith("cluster"))


def enclose_args(args):
    enclosed = []
    for a in args:
        key, value = a.split("=")
        enclosed.append(f'{key}="{value}"')
    return enclosed


def handle_thoth_gpu(cfg: DictConfig, hydra_cfg: DictConfig):
    """
    Output example:
    -----------------
    #!/usr/bin/zsh

    #OAR -l walltime=12:00:00
    #OAR -n mybigbeautifuljob
    #OAR -t besteffort
    #OAR -t idempotent
    #OAR -p gpumem>'20000'
    #OAR -p gpumodel='p100' #NOTE: last property takes priority
    #OAR -d /path/to/dir/
    #OAR -E /path/to/file.stderr
    #OAR -O /path/to/file.stdout

    source gpu_setVisibleDevices.sh

    conda activate $my_env

    python train.py $overrides

    """
    #########
    # Setup #
    #########
    cmd = ""
    cluster = cfg.cluster
    launcher = cfg.launcher
    # Create OAR log folder
    os.makedirs(f"{cluster.engine}", exist_ok=True)
    # Copy config file when job is created
    shutil.copy2(".hydra/config.yaml", "config.yaml")

    ###########
    # Command #
    ###########
    # Shebang
    cmd += f"#!{cluster.shell.bin_path}\n"
    # New line
    cmd += "\n"
    # NOTE: Use hours instead of complete walltime
    # Walltime
    cmd += f"{cluster.directive} -l walltime={launcher.hours}:00:00\n"
    # Filter out launcher/cluster args from config overrides
    overrides = hydra_cfg.overrides.task
    filtered_args = list(filter(filter_fn, overrides))
    job_name = ",".join([a.split(".")[-1] for a in filtered_args])
    # Limit job name length
    job_name = job_name[: min(50, len(job_name))]
    # Add job name
    cmd += f"{cluster.directive} -n {launcher.cmd}|{job_name}\n"
    # Best effort
    if launcher.besteffort:
        cmd += f"{cluster.directive} -t besteffort\n"
    if launcher.idempotent:
        cmd += f"{cluster.directive} -t idempotent\n"
    # GPU memory property
    # NOTE: units = Gb
    gpumem = f"{launcher.gpumem}000"
    if launcher.gpumem is not None:
        cmd += f"{cluster.directive} -p gpumem>{gpumem!r}\n"
    # GPU model property
    # NOTE: last property takes priority
    if launcher.gpumodel is not None:
        if type(launcher.gpumodel) == ListConfig:
            cmd += f"{cluster.directive} -p "
            cmd += " or ".join([f"gpumodel={m!r}" for m in launcher.gpumodel])
            cmd += "\n"
        else:
            cmd += f"{cluster.directive} -p gpumodel={launcher.gpumodel!r}\n"
    # # Path to dir
    cmd += f"{cluster.directive} -d {hydra_cfg.runtime.cwd}\n"
    # Job output files
    cwd = os.getcwd()
    # Stderr
    err_path = os.path.join(cwd, f"{cluster.engine}/%jobid%.stderr")
    cmd += f"{cluster.directive} -E {err_path}\n"
    # Stdout
    out_path = os.path.join(cwd, f"{cluster.engine}/%jobid%.stdout")
    cmd += f"{cluster.directive} -O {out_path}\n"
    # Newline
    cmd += "\n"
    # Print hostname and devices
    cmd += 'echo "Host is `hostname`"\n'
    # TODO devices
    # Shell instance
    cmd += f"source {cluster.shell.config_path}\n"
    # Newline
    cmd += "\n"
    # source gpu_setVisibleDevices.sh
    cmd += f"{cluster.cleanup}\n"
    # Newline
    cmd += "\n"
    # Conda environment
    cmd += f"conda activate {cluster.conda_env}\n"
    # Newline
    cmd += "\n"
    # Python command
    args = " ".join(enclose_args(filtered_args))
    cmd += f"python {launcher.cmd} {args} hydra.run.dir={cwd}"
    return cmd


def launch(cfg: DictConfig) -> None:

    logger.debug(cfg)
    cluster = cfg.cluster
    launcher = cfg.launcher
    assert (
        launcher.name in cluster.launchers
    ), f"{launcher.name} not in {cluster.launchers}"
    # Get Hydra Config
    hydra_cfg = HydraConfig.get()
    # Build command
    cmd = handle_thoth_gpu(cfg, hydra_cfg)

    logger.info(f"Selected cluster => {cluster.engine}")
    logger.info(f"Using {cluster.shell.bin_path!r} for shebang")
    logger.debug(cmd)

    # Path to script
    script_path = os.path.join(os.getcwd(), launcher.filename)
    # Write script file
    with open(script_path, "w") as f:
        f.write(cmd)

    # Make file executable
    chmod_cmd = f"chmod +x {script_path!r}"
    subprocess.check_call(chmod_cmd, shell=True)

    # Connect to frontal node and launch command
    launch_cmd = f'ssh {cluster.frontal_node} "{cluster.cmd} {script_path!r}"'
    logger.debug(launch_cmd)
    # Launch job over SSH
    subprocess.check_call(launch_cmd, shell=True)

    # Print job ID
    try:
        job_id = hydra_cfg.job.id
    except MissingMandatoryValue:
        job_id = "MAIN"

    logger.info(f"Job {job_id} launched!")


@hydra.main(config_path="hsi_unmixing/config", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info(f"Current working directory: {os.getcwd()}")
    try:
        launch(cfg)
    except Exception as e:
        logger.critical(e, exc_info=True)


if __name__ == "__main__":
    main()

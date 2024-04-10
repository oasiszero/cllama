import typer
import uuid
import shutil
from urllib.parse import urlparse
import yaml
import json
import psutil
import questionary
import os
import re
import subprocess
import pyaml
import pathlib
from cllama.spec import GPU_MEMORY


HOME_DIR = pathlib.Path.home()
CLLAMA_HOME = HOME_DIR / ".cllama"
BENTOML_HOME = CLLAMA_HOME / "bentoml"
REPO_DIR = CLLAMA_HOME / "repos"
CACHE_DIR = CLLAMA_HOME / "cache"

CLLAMA_HOME.mkdir(exist_ok=True, parents=True)
CACHE_DIR.mkdir(exist_ok=True, parents=True)
REPO_DIR.mkdir(exist_ok=True, parents=True)


app = typer.Typer()


"""
Usage:
  cllama [flags]
  cllama [command]

Available Commands:
  serve       Start ollama
  create      Create a model from a Modelfile
  show        Show information for a model
  run         Run a model
  pull        Pull a model from a registry
  push        Push a model to a registry
  list        List models
  cp          Copy a model
  rm          Remove a model
  help        Help about any command

Flags:
  -h, --help      help for ollama
  -v, --version   Show version information

Use "ollama [command] --help" for more information about a command.
"""


MODEL_INFOS = {
    "llama2": {
        "7b": "git+https://github.com/bojiang/bentovllm@main#subdirectory=llama2-7b-chat",
        "7b-chat": "git+https://github.com/bojiang/bentovllm@main#subdirectory=llama2-7b-chat",
        "7b-chat-fp16": "git+https://github.com/bojiang/bentovllm@main#subdirectory=llama2-7b-chat",
    },
}


# match repo, branch, subdirectory
REG_GITPACKAGE = r"git\+(?P<repo>.+?)@(?P<branch>.+?)#subdirectory=(?P<subdirectory>.+)"
REG_GITPACKAGE = re.compile(REG_GITPACKAGE)


@app.command()
def serve():
    pass


def _cli_install_aws():
    if subprocess.check_call(["which", "aws"]) != 0:
        print("aws cli is not installed")
    else:
        print("aws cli is not configured")
    return


def _filter_instance_types(
    instance_types,
    gpu_count,
    gpu_memory=None,
    gpu_type=None,
    level="match",
):
    if gpu_memory is None:
        if gpu_type is None:
            raise ValueError("Either gpu_memory or gpu_type must be provided")
        gpu_memory = GPU_MEMORY[gpu_type]

    def _check_instance(spec):
        if gpu_count == 0 or gpu_count is None:
            if "GpuInfo" in spec:
                return False
            else:
                return True
        else:
            gpus = spec.get("GpuInfo", {}).get("Gpus", [])
            if len(gpus) == 0:
                return False
            it_gpu = gpus[0]
            it_gpu_mem = it_gpu["MemoryInfo"]["SizeInMiB"] / 1024

            if it_gpu["Count"] == gpu_count and it_gpu_mem == gpu_memory:
                return True
            elif it_gpu["Count"] >= gpu_count and it_gpu_mem >= gpu_memory:
                if level == "match":
                    return False
                elif level == "usable":
                    return True
                else:
                    assert False
            else:
                return False

    def _sort_key(spec):
        return (
            spec["InstanceType"].split(".")[0],
            spec.get("GpuInfo", {}).get("Gpus", [{}])[0].get("Count", 0),
            spec.get("VCpuInfo", {}).get("DefaultVCpus", 0),
            spec.get("MemoryInfo", {}).get("SizeInMiB", 0),
        )

    return sorted(filter(_check_instance, instance_types), key=_sort_key)


def _get_bento_info(tag):
    cmd = ["bentoml", "get", tag]
    print(f"\n$ {' '.join(cmd)}")
    get_result = subprocess.check_output(
        cmd,
        env=dict(os.environ, BENTOML_HOME=BENTOML_HOME),
    )
    get_result = "\n".join(
        [line.split("!!")[0] for line in get_result.decode().splitlines()]
    )
    bento_info = yaml.safe_load(get_result)
    return bento_info


def _build_bento(bento_project_dir, _, tag):
    cmd = ["bentoml", "build", ".", "--version", tag, "--output", "tag"]
    print(f"\n$ {' '.join(cmd)}")
    build_result = subprocess.check_output(
        cmd,
        cwd=bento_project_dir,
        env=dict(os.environ, BENTOML_HOME=BENTOML_HOME),
    )
    tag = build_result.decode().strip()
    if tag.startswith("__tag__:"):
        tag = tag.strip("__tag__:")
    return tag.split(":")


def _resolve_git_package(package):
    match = REG_GITPACKAGE.match(package)
    if not match:
        raise ValueError(f"Invalid git package: {package}")
    repo_url, branch, subdirectory = match.groups()
    parsed = urlparse(repo_url)

    path_parts = [parsed.netloc] + parsed.path.split("/")

    return repo_url, branch, subdirectory, path_parts


def _get_it_card(spec):
    """
    InstanceType: g4dn.2xlarge
    VCpuInfo:
      DefaultCores: 32
      DefaultThreadsPerCore: 2
      DefaultVCpus: 64

    MemoryInfo:
      SizeInMiB: 32768

    GpuInfo:
      Gpus:
        - Count: 1
          Manufacturer: NVIDIA
          MemoryInfo:
            SizeInMiB: 16384
          Name: T4
      TotalGpuMemoryInMiB: 16384
    """
    return f"{spec['InstanceType']} (cpus: {spec['VCpuInfo']['DefaultVCpus']}, mem: {spec['MemoryInfo']['SizeInMiB']}, gpu: {spec['GpuInfo']['Gpus'][0]['Name']} x {spec['GpuInfo']['Gpus'][0]['Count']})"


def _ensure_aws_security_group(group_name="cllama-http-default"):
    try:
        existing_groups = subprocess.check_output(
            [
                "aws",
                "ec2",
                "describe-security-groups",
                "--filters",
                f"Name=group-name,Values={group_name}",
            ]
        )
        existing_groups = json.loads(existing_groups)
        if existing_groups["SecurityGroups"]:
            return existing_groups["SecurityGroups"][0]["GroupId"]

        result = subprocess.check_output(
            [
                "aws",
                "ec2",
                "create-security-group",
                "--group-name",
                group_name,
                "--description",
                "Default VPC security group for cllama services",
            ]
        )
        result = json.loads(result)
        security_group_id = result["GroupId"]

        subprocess.check_call(
            [
                "aws",
                "ec2",
                "authorize-security-group-ingress",
                "--group-id",
                security_group_id,
                "--protocol",
                "tcp",
                "--port",
                "80",
                "--cidr",
                "0.0.0.0/0",
            ]
        )
        subprocess.check_call(
            [
                "aws",
                "ec2",
                "authorize-security-group-ingress",
                "--group-id",
                security_group_id,
                "--protocol",
                "tcp",
                "--port",
                "443",
                "--cidr",
                "0.0.0.0/0",
            ]
        )
        subprocess.check_call(
            [
                "aws",
                "ec2",
                "authorize-security-group-ingress",
                "--group-id",
                security_group_id,
                "--protocol",
                "tcp",
                "--port",
                "22",
                "--cidr",
                "0.0.0.0/0",
            ]
        )
        return security_group_id
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to create security group: {e}")


@app.command()
def run(model: str, tag: str = "latest", force_rebuild: bool = False):
    if tag == "latest":
        tag = next(iter(MODEL_INFOS[model].keys()))

    package = MODEL_INFOS[model][tag]
    repo, branch, subdirectory, path_parts = _resolve_git_package(package)
    try:
        bento_info = _get_bento_info(f"{model}:{tag}")
    except subprocess.CalledProcessError:
        repo_dir = REPO_DIR.joinpath(*path_parts)

        if force_rebuild:
            shutil.rmtree(repo_dir, ignore_errors=True)

        if not repo_dir.exists():
            repo_dir.parent.mkdir(parents=True, exist_ok=True)
            try:
                cmd = ["git", "clone", "--branch", branch, repo, str(repo_dir)]
                print(f"\n$ {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            except:
                shutil.rmtree(repo_dir, ignore_errors=True)
                raise

        bento_project_dir = repo_dir / subdirectory
        bento, tag = _build_bento(bento_project_dir, model, tag)
        bento_info = _get_bento_info(f"{bento}:{tag}")

    if len(bento_info["services"]) != 1:
        raise ValueError("Only support one service currently")

    cloud_provider = questionary.select(
        "Select a cloud provider",
        choices=["aws", "gcp(not supported)", "azure(not supported))"],
    ).ask()

    if cloud_provider == "aws":
        try:
            cmd = ["aws", "ec2", "describe-instance-types"]
            print(f"\n$ {' '.join(cmd)}")
            _instance_types = subprocess.check_output(cmd, text=True)
        except subprocess.CalledProcessError:
            raise
            # print(e)
            # _cli_install_aws()
        available_it_infos = json.loads(_instance_types)["InstanceTypes"]
        # pyaml.p(available_it_infos)

        service = bento_info["services"][0]
        if "config" not in service or "resources" not in service["config"]:
            raise ValueError("Service config is missing")
        elif "gpu" in service["config"]["resources"]:
            gpu_count = service["config"]["resources"]["gpu"]
            gpu_type = service["config"]["resources"].get("gpu_type")
            gpu_memory = service["config"]["resources"].get("gpu_memory")
            supported_its = _filter_instance_types(
                available_it_infos,
                gpu_count,
                gpu_memory,
                gpu_type,
            )
            it = questionary.select(
                "Select an instance type",
                choices=[
                    questionary.Choice(
                        title=_get_it_card(it_info),
                        value=it_info["InstanceType"],
                    )
                    for it_info in supported_its
                ],
            ).ask()
            security_group_id = _ensure_aws_security_group()
            AMI = "ami-02623cf022763d4a1"

            init_script_file = CACHE_DIR / f"init_script_{str(uuid.uuid4())[:8]}.sh"
            with open(init_script_file, "w") as f:
                f.write(
                    INIT_SCRIPT_TEMPLATE.format(
                        repo=repo,
                        subdirectory=subdirectory,
                    )
                )
            cmd = [
                "aws",
                "ec2",
                "run-instances",
                "--image-id",
                AMI,
                "--instance-type",
                it,
                "--security-group-ids",
                security_group_id,
                "--user-data",
                f"file://{init_script_file}",
                "--key-name",
                "jiang",
                "--count",
                "1",
            ]
            print(f"\n$ {' '.join(cmd)}")

        else:
            raise ValueError("GPU is required for now")


INIT_SCRIPT_TEMPLATE = """pip3 install bentoml
git clone {repo} bento_repo
cd bento_repo/{subdirectory}
pip3 install -r requirements.txt
/home/ubuntu/.local/bin/bentoml serve . --port 80
"""


@app.command()
def list():
    pyaml.p({"models": {k: tuple(v.keys()) for k, v in MODEL_INFOS.items()}})


if __name__ == "__main__":
    app()

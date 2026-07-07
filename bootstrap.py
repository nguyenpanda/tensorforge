#!/usr/bin/env python3
"""
TensorForge Multi-Platform Bootstrap Script
Handles OS verification, NFS I/O bottleneck resolution via /tmp symlinking,
interactive and non-interactive learning path selection, and dynamic PyTorch CUDA/MPS URL resolution.

Uses standard libraries at module load time for clean idempotency before environment synchronization.
"""

from __future__ import annotations

import argparse
import contextlib
import getpass
import os
import platform
import re
import shutil
import subprocess
import sys
import time

try:
    from nguyenpanda.swan import RESET as SWAN_RESET
    from nguyenpanda.swan import c8, c24, color

    RESET = SWAN_RESET
    CYAN_LOGO = c24[(0, 220, 255)]
    MAGENTA_LINE = c24[(180, 80, 255)]
    WHITE_TITLE = c24[(255, 255, 255)]
    HEADER_CYAN = color["c"]
    INFO_BLUE = color["b"]
    SUCCESS_GREEN = color["g"]
    WARN_YELLOW = color["y"]
    ERROR_RED = color["r"]
    PROMPT_YELLOW = color["y"]
    BRACKET_CYAN = color["c"]
    CMD_GREEN = color["g"]
    SEP_CYAN = c8[45]
except ImportError:
    RESET = ""
    CYAN_LOGO = ""
    MAGENTA_LINE = ""
    WHITE_TITLE = ""
    HEADER_CYAN = ""
    INFO_BLUE = ""
    SUCCESS_GREEN = ""
    WARN_YELLOW = ""
    ERROR_RED = ""
    PROMPT_YELLOW = ""
    BRACKET_CYAN = ""
    CMD_GREEN = ""
    SEP_CYAN = ""


def print_banner() -> None:
    print(f"\n{MAGENTA_LINE}{'=' * 72}{RESET}")
    print(
        f"{CYAN_LOGO}  ████████╗███████╗███╗   ██╗███████╗██████╗ ██████╗  ██████╗ ███████╗███████╗{RESET}"
    )
    print(
        f"{CYAN_LOGO}  ╚══██╔══╝██╔════╝████╗  ██║██╔════╝██╔═══██╗██╔══██╗██╔══██╗██╔═══██╗██╔══██╗██╔════╝{RESET}"
    )
    print(
        f"{CYAN_LOGO}     ██║   █████╗  ██╔██╗ ██║███████╗██║   ██║██████╔╝██████╔╝██║   ██║██████╔╝█████╗  {RESET}"
    )
    print(
        f"{CYAN_LOGO}     ██║   ██╔══╝  ██║╚██╗██║╚════██║██║   ██║██╔══██╗██╔══██╗██║   ██║██╔══██╗██╔══╝  {RESET}"
    )
    print(
        f"{CYAN_LOGO}     ██║   ███████╗██║ ╚████║███████║╚██████╔╝██║  ██║██║  ██║╚██████╔╝██║  ██║███████╗{RESET}"
    )
    print(
        f"{CYAN_LOGO}     ╚═╝   ╚══════╝╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝{RESET}"
    )
    print(f"{MAGENTA_LINE}{'=' * 72}{RESET}")
    print(
        f"{WHITE_TITLE}          TensorForge Multi-Platform Bootstrap & Environment Pipeline{RESET}\n"
    )


def print_header(title: str) -> None:
    print(f"\n{HEADER_CYAN}{'=' * 60}{RESET}")
    print(f"{HEADER_CYAN}  {title}{RESET}")
    print(f"{HEADER_CYAN}{'=' * 60}{RESET}\n")


def print_info(msg: str) -> None:
    print(f"{INFO_BLUE}[INFO]{RESET} {msg}")


def print_success(msg: str) -> None:
    print(f"{SUCCESS_GREEN}[SUCCESS]{RESET} {msg}")


def print_warn(msg: str) -> None:
    print(f"{WARN_YELLOW}[WARNING]{RESET} {msg}")


def print_error(msg: str) -> None:
    print(f"{ERROR_RED}[ERROR]{RESET} {msg}")


def prompt_yes_no(question: str, default: bool = False, non_interactive: bool = False) -> bool:
    if non_interactive:
        return default
    default_str = "y/N" if not default else "Y/n"
    while True:
        try:
            prompt_str = f"{PROMPT_YELLOW}{question} [{default_str}]: {RESET}"
            ans = input(prompt_str).strip().lower()
            if not ans:
                return default
            if ans in ("y", "yes"):
                return True
            if ans in ("n", "no"):
                return False
            print_warn("Please enter 'y' or 'n'.")
        except (KeyboardInterrupt, EOFError):
            print()
            print_info(
                f"No input detected or cancelled. Defaulting to: {'yes' if default else 'no'}"
            )
            return default


def prompt_choice(
    question: str, options: list[str], default_idx: int = 1, non_interactive: bool = False
) -> int:
    if non_interactive:
        print_info(
            f"Non-interactive mode: auto-selecting option [{default_idx}]: {options[default_idx - 1]}"
        )
        return default_idx
    print(f"\n{PROMPT_YELLOW}{question}{RESET}")
    for idx, opt in enumerate(options, 1):
        print(f"  {BRACKET_CYAN}[{idx}]{RESET} {opt}")
    while True:
        try:
            ans = input(
                f"{PROMPT_YELLOW}Select an option [1-{len(options)}] (default {default_idx}): {RESET}"
            ).strip()
            if not ans:
                print_info(f"Defaulting to option [{default_idx}]: {options[default_idx - 1]}")
                return default_idx
            if ans.isdigit():
                val = int(ans)
                if 1 <= val <= len(options):
                    return val
            print_warn(f"Please enter a number between 1 and {len(options)}.")
        except (KeyboardInterrupt, EOFError):
            print()
            print_info(
                f"No input detected or cancelled. Defaulting to option [{default_idx}]: {options[default_idx - 1]}"
            )
            return default_idx


def wipe_venv() -> None:
    venv_path = os.path.abspath("./.venv")
    if os.path.exists(venv_path) or os.path.islink(venv_path):
        print_info(f"Wiping existing virtual environment at: {venv_path}")
        if os.path.islink(venv_path):
            try:
                target = os.readlink(venv_path)
                os.unlink(venv_path)
                if os.path.exists(target):
                    shutil.rmtree(target, ignore_errors=True)
            except Exception as e:
                print_warn(f"Error unlinking virtual environment: {e}")
        shutil.rmtree(venv_path, ignore_errors=True)
        if os.path.exists(venv_path) or os.path.islink(venv_path):
            with contextlib.suppress(Exception):
                os.remove(venv_path)
        print_success("Successfully wiped .venv for a clean reconfiguration.")


def phase1_os_sanity_check() -> None:
    print_header("Phase 1: OS & Python Sanity Check")

    current_os = platform.system()
    if current_os == "Windows" or os.name == "nt":
        print_error("Windows is explicitly not supported. Please use WSL2, Linux, or macOS.")
        sys.exit(1)

    print_success(f"Supported OS detected: {current_os} ({platform.machine()})")

    py_ver = sys.version_info
    if py_ver < (3, 12):
        print_error(
            f"Python >= 3.12 is required. Detected Python {py_ver.major}.{py_ver.minor}.{py_ver.micro}"
        )
        sys.exit(1)

    print_success(f"Python version check passed: {py_ver.major}.{py_ver.minor}.{py_ver.micro}")


def phase2_nfs_check(non_interactive: bool = False) -> None:
    print_header("Phase 2: Storage & NFS Bottleneck Check")

    is_nfs = prompt_yes_no(
        "Are you running this project on a shared network drive (NFS)?",
        default=False,
        non_interactive=non_interactive,
    )
    if is_nfs:
        print_info("A local '.venv' on NFS causes degraded I/O and hardlinking failures with 'uv'.")
        venv_path = os.path.abspath("./.venv")

        if os.path.exists(venv_path) or os.path.islink(venv_path):
            if os.path.islink(venv_path):
                target = os.readlink(venv_path)
                print_info(f"Existing symlink detected at ./venv -> {target}")
                if prompt_yes_no(
                    "Do you want to remove the existing symlink and recreate it?",
                    default=True,
                    non_interactive=non_interactive,
                ):
                    os.unlink(venv_path)
                    print_success("Removed existing .venv symlink.")
            elif os.path.isdir(venv_path):
                print_warn(f"Existing physical directory detected at {venv_path}.")
                if prompt_yes_no(
                    "Do you want to safely remove ./venv via atomic rename and background deletion to eliminate NFS bottlenecks?",
                    default=True,
                    non_interactive=non_interactive,
                ):
                    trash_dir_name = f"trash_venv_{int(time.time())}"
                    print_info(f"Atomically moving ./venv -> {trash_dir_name}...")
                    os.rename(venv_path, trash_dir_name)
                    print_info("Spawning detached background process to remove trash directory...")
                    subprocess.Popen(["rm", "-rf", trash_dir_name])
                    print_success(
                        "Successfully moved ./venv to trash for asynchronous background deletion."
                    )
            else:
                if prompt_yes_no(
                    "Existing ./venv file detected. Remove it?",
                    default=True,
                    non_interactive=non_interactive,
                ):
                    os.remove(venv_path)

        username = getpass.getuser()
        tmp_venv_dir = f"/tmp/tforge_venv_{username}"

        print_info(f"Creating target virtual environment directory at: {tmp_venv_dir}")
        os.makedirs(tmp_venv_dir, exist_ok=True)

        if not os.path.exists(venv_path) and not os.path.islink(venv_path):
            print_info(f"Creating secure symlink: {venv_path} -> {tmp_venv_dir}")
            try:
                os.symlink(tmp_venv_dir, venv_path)
                print_success("Symlink created successfully.")
            except OSError as e:
                print_error(f"Failed to create symlink: {e}")
                sys.exit(1)
        else:
            print_info(f"Using existing ./venv at: {venv_path}")
    else:
        print_success("Local storage detected. Proceeding with default ./venv configuration.")


def phase3_module_selection(
    target_path: str | None = None, non_interactive: bool = False
) -> tuple[int, list[str], str]:
    print_header("Phase 3: Module & Hardware Selection Menu")

    if target_path:
        if target_path == "core":
            choice = 1
        elif target_path == "torch":
            choice = 2
        elif target_path in ("hpc", "all"):
            choice = 4
        elif target_path == "dev":
            choice = 1
        else:
            choice = 4
        print_info(f"Using CLI specified target path: '{target_path}' -> path [{choice}]")
    else:
        options = [
            "Core (Numpy only - `arraysmith`) -> extras: dev",
            "AI Models (PyTorch - `tensorsmith`) -> extras: torch, dev",
            "HPC Kernels (PyTorch + Compilers - `hpcsmith`) -> extras: torch, hpc, dev",
            "Install Everything -> extras: torch, hpc, dev",
        ]
        choice = prompt_choice(
            "Select your learning path:", options, default_idx=4, non_interactive=non_interactive
        )

    if choice == 1:
        extras = ["dev"]
        extras_display = "dev"
    elif choice == 2:
        extras = ["torch", "dev"]
        extras_display = "torch, dev"
    elif choice in (3, 4):
        extras = ["torch", "hpc", "dev"]
        extras_display = "torch, hpc, dev"
    else:
        extras = ["torch", "hpc", "dev"]
        extras_display = "torch, hpc, dev"

    print_success(f"Selected learning path [{choice}]. Extras to install: {extras_display}")
    return choice, extras, extras_display


def phase4_dynamic_url_resolution(choice: int, non_interactive: bool = False) -> str:
    print_header("Phase 4: Dynamic PyTorch URL Resolution")

    if choice == 1:
        print_info(
            "Core learning path selected (No PyTorch needed). Skipping dynamic URL resolution."
        )
        return ""

    current_os = platform.system()
    index_args = ""

    if current_os == "Darwin":
        arch = platform.machine().lower()
        if arch in ("arm64", "aarch64"):
            print_success(
                "Apple Silicon detected. Native MPS support enabled via default PyPI wheels."
            )
        elif arch in ("x86_64", "amd64", "i386", "i686"):
            print_warn("Apple Intel detected. CPU-only fallback enabled.")
        else:
            print_info(f"macOS ({arch}) detected. Using default PyPI wheels.")
        index_args = ""
    elif current_os == "Linux":
        default_choice = 5
        print_info("Checking for NVIDIA GPU via 'nvidia-smi'...")
        try:
            res = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            if res.returncode == 0:
                print_success("NVIDIA GPU detected via nvidia-smi:")
                for line in res.stdout.splitlines():
                    if any(
                        key in line
                        for key in (
                            "NVIDIA",
                            "Driver Version",
                            "RTX",
                            "GTX",
                            "Tesla",
                            "A100",
                            "H100",
                            "L40",
                        )
                    ):
                        print_info(f"  {line.strip()}")

                cuda_match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", res.stdout)
                driver_match = re.search(r"Driver Version:\s*(\d+)\.(\d+)", res.stdout)
                if cuda_match:
                    major = int(cuda_match.group(1))
                    minor = int(cuda_match.group(2))
                    if (major, minor) >= (12, 6):
                        default_choice = 1
                    elif (major, minor) >= (12, 4):
                        default_choice = 2
                    elif (major, minor) >= (12, 0):
                        default_choice = 3
                    elif (major, minor) >= (11, 0):
                        default_choice = 4
                    else:
                        default_choice = 1
                elif driver_match:
                    driver_major = int(driver_match.group(1))
                    if driver_major >= 550:
                        default_choice = 1
                    elif driver_major >= 525:
                        default_choice = 2
                    elif driver_major >= 450:
                        default_choice = 4
                    else:
                        default_choice = 1
                else:
                    default_choice = 1
            else:
                print_warn(
                    "nvidia-smi command returned non-zero exit code. No active CUDA GPU detected."
                )
        except FileNotFoundError:
            print_warn("'nvidia-smi' not found in PATH. No CUDA GPU detected.")
        except Exception as e:
            print_warn(f"Could not execute nvidia-smi: {e}")

        options = [
            "CUDA 12.6",
            "CUDA 12.4",
            "CUDA 12.1",
            "CUDA 11.8",
            "CPU Only",
        ]
        if 1 <= default_choice <= 4:
            options[default_choice - 1] += " (Recommended for detected native driver)"
        elif default_choice == 5:
            options[4] += " (Recommended for CPU-only nodes)"

        linux_choice = prompt_choice(
            "Select target PyTorch environment for Linux:",
            options,
            default_idx=default_choice,
            non_interactive=non_interactive,
        )

        if linux_choice == 1:
            index_args = "--extra-index-url https://download.pytorch.org/whl/cu126"
        elif linux_choice == 2:
            index_args = "--extra-index-url https://download.pytorch.org/whl/cu124"
        elif linux_choice == 3:
            index_args = "--extra-index-url https://download.pytorch.org/whl/cu121"
        elif linux_choice == 4:
            index_args = "--extra-index-url https://download.pytorch.org/whl/cu118"
        elif linux_choice == 5:
            index_args = "--extra-index-url https://download.pytorch.org/whl/cpu"
        else:
            index_args = "--extra-index-url https://download.pytorch.org/whl/cu126"

        print_success(
            f"Configured PyTorch index argument: {index_args if index_args else '(default PyPI)'}"
        )
    else:
        print_info(f"OS '{current_os}' detected. Using default PyPI index.")
        index_args = ""

    return index_args


def print_post_sync_topology_summary() -> None:
    import glob

    for p in glob.glob(os.path.abspath("./.venv/lib/python*/site-packages")):
        if p not in sys.path:
            sys.path.insert(0, p)

    try:
        from nguyenpanda.swan import RESET as SWAN_RESET
        from nguyenpanda.swan import c8, c24, color

        color["c"]
        info_b = color["b"]
        success_g = color["g"]
        reset = SWAN_RESET
        accent = c24[(0, 220, 255)]
        box_c = c8[45]
    except ImportError:
        info_b = INFO_BLUE
        success_g = SUCCESS_GREEN
        reset = RESET
        accent = CYAN_LOGO
        box_c = SEP_CYAN

    print(
        f"\n{box_c}╔═════════════════════════════════════════════════════════════════════╗{reset}"
    )
    print(
        f"{box_c}║{reset}  {accent}Hardware Topology & Environment Summary{reset}                        {box_c}║{reset}"
    )
    print(f"{box_c}╠═════════════════════════════════════════════════════════════════════╣{reset}")

    os_name = f"{platform.system()} ({platform.machine()})"
    cpu_cores = str(os.cpu_count() or "Unknown")

    try:
        if platform.system() == "Darwin":
            mem_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
            ram_gb = f"{mem_bytes / (1024**3):.1f} GB"
        elif platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        ram_gb = f"{kb / (1024**2):.1f} GB"
                        break
                else:
                    ram_gb = "Unknown"
        else:
            ram_gb = "Unknown"
    except Exception:
        ram_gb = "Unknown"

    gpu_info = "None (CPU only)"
    if platform.system() == "Darwin" and platform.machine().lower() in ("arm64", "aarch64"):
        gpu_info = "Apple Silicon MPS (Unified Memory)"
    elif platform.system() == "Linux":
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
            )
            if res.returncode == 0 and res.stdout.strip():
                gpu_info = f"NVIDIA {res.stdout.strip()}"
        except Exception:
            pass

    print(f"{box_c}║{reset}  • {info_b}Operating System:{reset}  {os_name:<45} {box_c}║{reset}")
    print(f"{box_c}║{reset}  • {info_b}CPU Cores:{reset}         {cpu_cores:<45} {box_c}║{reset}")
    print(f"{box_c}║{reset}  • {info_b}System Memory:{reset}     {ram_gb:<45} {box_c}║{reset}")
    print(f"{box_c}║{reset}  • {info_b}GPU / Accelerator:{reset} {gpu_info:<45} {box_c}║{reset}")
    print(
        f"{box_c}║{reset}  • {info_b}Python Version:{reset}    {sys.version.split()[0]:<45} {box_c}║{reset}"
    )
    print(
        f"{box_c}║{reset}  • {info_b}Runtime Path:{reset}      {sys.executable:<45} {box_c}║{reset}"
    )
    print(
        f"{box_c}╚═════════════════════════════════════════════════════════════════════╝{reset}\n"
    )
    print(
        f"{success_g}[SUCCESS]{reset} Environment is ready! Run 'tforge validate' to verify baseline integrity.\n"
    )


def phase5_execution(extras: list[str], extras_display: str, index_args: str) -> None:
    print_header("Phase 5: Environment Synchronization & Execution")

    cmd_display = f"uv sync --extra {extras_display}"
    if index_args:
        cmd_display += f" {index_args}"

    print_info(f"Final bootstrap command: {CMD_GREEN}{cmd_display}{RESET}")

    env = os.environ.copy()
    env["UV_LINK_MODE"] = "copy"
    print_info(
        "Set environment variable: UV_LINK_MODE=copy (suppresses hardlink warnings across NFS/filesystems)"
    )

    cmd_list = ["uv", "sync"]
    for extra in extras:
        cmd_list.extend(["--extra", extra])
    if index_args:
        cmd_list.extend(index_args.split())

    print_info("Executing sync process via subprocess...\n")
    print(f"{SEP_CYAN}{'-' * 60}{RESET}")

    try:
        res = subprocess.run(cmd_list, env=env)
        print(f"{SEP_CYAN}{'-' * 60}{RESET}\n")
        if res.returncode != 0:
            print_error(f"Initialization command failed with exit code {res.returncode}.")
            sys.exit(res.returncode)
        else:
            print_success("TensorForge environment initialization completed successfully!")
            print_post_sync_topology_summary()
            print_info(
                "You can now run 'tforge validate' or 'tforge check all' to verify your workspace."
            )
    except FileNotFoundError:
        print_error(
            "'uv' executable not found. Please ensure Astral 'uv' is installed and in your PATH."
        )
        sys.exit(1)
    except Exception as e:
        print_error(f"An unexpected error occurred during execution: {e}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="TensorForge Multi-Platform Bootstrap Script")
    parser.add_argument(
        "--path",
        choices=["core", "torch", "hpc", "all", "dev"],
        help="Target learning path to install",
    )
    parser.add_argument(
        "--reconfigure",
        action="store_true",
        help="Wipe existing .venv before syncing",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Run non-interactively with default options",
    )
    args = parser.parse_args()

    if args.reconfigure:
        wipe_venv()

    print_banner()
    phase1_os_sanity_check()
    phase2_nfs_check(non_interactive=args.yes or bool(args.path))
    choice, extras, extras_display = phase3_module_selection(
        target_path=args.path, non_interactive=args.yes or bool(args.path)
    )
    index_args = phase4_dynamic_url_resolution(choice, non_interactive=args.yes or bool(args.path))
    phase5_execution(extras, extras_display, index_args)


if __name__ == "__main__":
    main()

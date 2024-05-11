set shell := ["zsh", "-cu"]

# Export all just variables as environment variables.
set export

WANDB_MODE := "offline"
WANDB_SILENT := "true"
HYDRA_FULL_ERROR := "1"

breakout:
    . ./.venv/bin/activate
    python src/main.py env.train.id=BreakoutNoFrameskip-v4 

crafter:
    . ./.venv/bin/activate
    python src/main.py env.train.id=CrafterReward-v1


craftax:
    . ./.venv/bin/activate
    # https://github.dev/MichaelTMatthews/Craftax/blob/fbe4b50b985d980ea2035aa046174fb069d0cffe/craftax/craftax_env.py#L19
    # python src/main.py env.train.id=Craftax-Pixels-AutoReset-v1
    python -m pdb src/main.py env.train.id=Craftax-Symbolic-AutoReset-v1
    

# minihack:
#     python src/main.py env.train.id=MiniHack-River-v0

# watch the latest runs
watch_latest:
    . ./.venv/bin/activate
    cd ./outputs && \
    cd *([-1]) && \
    cd *([-1]) && \
    scripts/play.sh -e -r -h


resume_latest:
    . ./.venv/bin/activate
    cd ./outputs && \
    cd *([-1]) && \
    cd *([-1]) && \
    scripts/resume.sh

default: 
    just --list

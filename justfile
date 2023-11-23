set shell := ["zsh", "-cu"]

breakout:
    python src/main.py env.train.id=BreakoutNoFrameskip-v4 

crafter:
    python src/main.py env.train.id=CrafterReward-v1

minihack:
    python src/main.py env.train.id=MiniHack-River-v0

# watch the latest runs
watch_latest:
    . ./.venv/bin/activate
    cd ./outputs && \
    cd *([-1]) && \
    cd *([-1]) && \
    scripts/play.sh -e -r -h

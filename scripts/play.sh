#!/usr/bin/bash

fps=15
header=0
reconstruction=0
save_mode=0
mode="agent_in_env"

while [ "$1" != "" ]; do
    case $1 in
        -f | --fps )
            shift
            fps=$1
            ;;
        -h | --header )
            header=1
            ;; # adds banner with env metadata like action
        -r | --reconstruction )
            reconstruction=1
            ;; # 3 panes [original_obs, resized_obs, reconstructed], doesn't do anything if any of -w -a or -e are set. shows quality of encoder decoder
        -s | --save-mode )
            save_mode=1
            ;; # lets you save the episode to mp4
        -a | --agent-world-model )
            mode="agent_in_world_model"
            ;; # the agent plays in the world model env, shows the quality of the dynamics model
        -e | --episode )
            mode="episode_replay"
            ;; # replay train, test, or imagined episodes. shows quality of dynamics model
        -w | --world-model )
            mode="play_in_world_model"
            ;; # human plays in world model
        * )
            echo Invalid usage : $1
            exit 1
    esac
    shift
done

python -m pdb src/play.py hydra.run.dir=. hydra.output_subdir=null +mode="${mode}" +fps="${fps}" +header="${header}" +reconstruction="${reconstruction}" +save_mode="${save_mode}"

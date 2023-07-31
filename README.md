# Joint Synthesis of Safety Certificate and Safe Control Policy Using Constrained Reinforcement Learning

This is the reimplementation with tensorflow 1.x based on OpenAI safety-starter-agents.

## Installation and Dependencies
Please refer to [OpenAI safety-starter-agents](https://github.com/openai/safety-starter-agents) and [OpenAI safety gym](https://github.com/openai/safety-gym) for installation first.

For tensorflow 1.x, you could use
```[bash]
pip install tensorflow==1.15
```
to install. Note that tensorflow 1.x does not support python 3.8 or greater.

Then 
```[bash]
cd ${project_root_dir}
pip install -e .
```

## Running the experiment
```[bash]
python3 ${project_root_dir}/scripts/experiment.py
```

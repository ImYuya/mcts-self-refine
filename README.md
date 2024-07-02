# MCTS-Self-Refine: Monte Carlo Tree Search for Question Answering

## Overview

MCTS-Self-Refine is a Python project that uses the Monte Carlo Tree Search (MCTS) algorithm to explore optimal answers to given questions. By utilizing large language models (LLM) to generate and evaluate responses, it aims to find high-quality answers.

## Features

- Answer optimization using the MCTS algorithm
- Efficient answer generation through parallel processing
- Visualization of the exploration process in Mermaid format
- Checkpoint functionality for saving and resuming sessions

## Requirements

- Python 3.11
- [Rye](https://rye-up.com/) (Package management tool)
- [Ollama](https://ollama.ai/) (Locally running LLM server)

## Setup

### Installing Ollama (macOS)

1. Install Ollama using Homebrew:

   ```
   brew install ollama
   ```

   If Homebrew is not installed, follow the instructions on the [Homebrew official website](https://brew.sh/) to install it.

2. Start Ollama:

   ```
   ollama serve
   ```

3. Download the required model (e.g., gemma2):

   ```
   ollama pull gemma2
   ```

   Note: Downloading the model may take some time, ranging from a few minutes to tens of minutes depending on your network speed.

### Project Setup

1. Clone the repository:

   ```
   git clone https://github.com/ImYuya/mcts-self-refine.git
   cd mcts-self-refine
   ```

2. Install dependencies using Rye:

   ```
   rye sync
   ```

   Note: Rye will automatically set up a Python 3.11 environment for the project.

## Usage

1. Activate the Rye virtual environment:

   ```
   rye shell
   ```

2. Run the script:

   ```
   rye run python src/mcts-self-refine-ollama-instructor.py
   ```

3. When prompted, enter your question.

4. The MCTS algorithm will run and generate the optimal answer.

5. The results will be displayed in the standard output, and the exploration tree visualization will be saved in the `mcts_tree_visualization.md` file in the `./output` folder.

## Customization

- Adjust the parameters of the `MCTS` class in the `src/mcts-self-refine-ollama-instructor.py` file to change the depth of exploration or the number of iterations.
- Modify the `model` parameter in the `LLM` class to use different LLM models.

## Troubleshooting

- Ensure that the Ollama server is running. If not, start it by running `ollama serve` in the terminal.
- If you encounter network connection issues, check your firewall settings.
- For problems with installing or using Ollama, refer to the [Ollama official documentation](https://github.com/ollama/ollama).
- Make sure Python 3.11 is installed on your system. While Rye manages the project's Python environment, Python 3.11 needs to be installed on your system.

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Contact

If you have any questions or suggestions, please [open an issue](https://github.com/ImYuya/mcts-self-refine/issues) or contact us through the project's GitHub page.

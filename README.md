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

6. The output markdown file (`mcts_tree_visualization.md`) will contain:
   - The initial question
   - A Mermaid diagram representing the MCTS tree
   - The best answer found by the algorithm

   Example of the output:

   # MCTS Tree Visualization

   Initial question: What are the key factors contributing to climate change, and what are the most effective strategies to mitigate its impact?

   ```mermaid
   graph TD
   0["Node 0<br>Depth: 0<br>What are the key fac...<br>Visits: 20<br>Value: 17.28"]
   style fill:#E6F3FF stroke:#4169E1 stroke-width:2px 0
   0 --> 1
   1["Node 1<br>Depth: 1<br>Climate change is a ...<br>Visits: 6<br>Value: 5.10"]
   style fill:#f9f stroke:#333 stroke-width:2px 1
   1 --> 4
   4["Node 4<br>Depth: 2<br>Climate change is a ...<br>Visits: 2<br>Value: 1.70"]
   style fill:#f9f stroke:#333 stroke-width:2px 4
   1 --> 5
   5["Node 5<br>Depth: 2<br>Climate change is a ...<br>Visits: 2<br>Value: 1.70"]
   style fill:#f9f stroke:#333 stroke-width:2px 5
   1 --> 6
   6["Node 6<br>Depth: 2<br>Climate change is a ...<br>Visits: 1<br>Value: 0.85"]
   style fill:#f9f stroke:#333 stroke-width:2px 6
   0 --> 2
   2["Node 2<br>Depth: 1<br>Climate change is a ...<br>Visits: 6<br>Value: 5.13"]
   style fill:#f9f stroke:#333 stroke-width:2px 2
   2 --> 10
   10["Node 10<br>Depth: 2<br>Climate change is a ...<br>Visits: 2<br>Value: 1.70"]
   style fill:#f9f stroke:#333 stroke-width:2px 10
   2 --> 11
   11["Node 11<br>Depth: 2<br>Climate change is a ...<br>Visits: 1<br>Value: 0.85"]
   style fill:#f9f stroke:#333 stroke-width:2px 11
   2 --> 12
   12["Node 12<br>Depth: 2<br>Climate change is a ...<br>Visits: 1<br>Value: 0.85"]
   style fill:#f9f stroke:#333 stroke-width:2px 12
   0 -->| stroke:#4169E1 stroke-width:2px| 3
   3["Node 3<br>Depth: 1<br>Climate change is a ...<br>Visits: 7<br>Value: 6.20"]
   style fill:#90EE90 stroke:#006400 stroke-width:4px 3
   3 --> 7
   7["Node 7<br>Depth: 2<br>Climate change is a ...<br>Visits: 2<br>Value: 1.80"]
   style fill:#f9f stroke:#333 stroke-width:2px 7
   3 --> 8
   8["Node 8<br>Depth: 2<br>Climate change is a ...<br>Visits: 2<br>Value: 1.80"]
   style fill:#f9f stroke:#333 stroke-width:2px 8
   3 --> 9
   9["Node 9<br>Depth: 2<br>Climate change is a ...<br>Visits: 2<br>Value: 1.75"]
   style fill:#f9f stroke:#333 stroke-width:2px 9
   ```

   ## Best Answer (Node 3)

   Climate change is a complex...

   (Note: The full answer provides detailed explanations of climate change factors and mitigation strategies.)

   This markdown file can be viewed in any markdown-compatible viewer or editor, and the Mermaid diagram will be automatically rendered on GitHub to visualize the MCTS tree structure.

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
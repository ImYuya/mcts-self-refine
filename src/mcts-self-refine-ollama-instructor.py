import math
import json
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field
import instructor
from openai import OpenAI
import random
import os

# ログ出力を抑制
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

class EvaluationCriteria(BaseModel):
    relevance: float = Field(..., description="Relevance to the question (0-1)")
    accuracy: float = Field(..., description="Accuracy of information (0-1)")
    depth: float = Field(..., description="Depth of explanation (0-1)")
    clarity: float = Field(..., description="Clarity of expression (0-1)")
    originality: float = Field(..., description="Originality of insights (0-1)")
    overall_score: float = Field(..., description="Overall score (0-1)")

class Node:
    def __init__(self, state: str, parent=None, depth=0):
        self.state = state
        self.parent = parent
        self.children: List[Node] = []
        self.visits = 0
        self.value = 0
        self.id = random.randint(1000, 9999)
        self.depth = depth

    def to_dict(self):
        return {
            'state': self.state,
            'visits': self.visits,
            'value': self.value,
            'children': [child.to_dict() for child in self.children],
            'depth': self.depth
        }

class MCTS:
    def __init__(self, llm, C=1.41, max_iterations=20, exploration_factor=0.15, max_depth=2):
        self.llm = llm
        self.C = C
        self.max_iterations = max_iterations
        self.exploration_factor = exploration_factor
        self.node_count = 0
        self.max_depth = max_depth

    def search(self, root_state: str) -> str:
        root = Node(root_state, depth=0)
        root.id = self.node_count
        self.node_count += 1

        print(f"Initial question: {root_state}")
        print(f"\nStarting MCTS search with max depth: {self.max_depth}")

        for i in range(self.max_iterations):
            print(f"\nIteration {i+1}/{self.max_iterations}")
            node = self.select(root)
            print(f"Selected node (depth {node.depth}): {node.state[:50]}...")
            
            if node.visits == 0 and node.depth < self.max_depth:
                self.expand(node)
                print(f"Expanded node with {len(node.children)} new children")
            
            if node.children:
                child = self.select(node)
                print(f"Selected child (depth {child.depth}): {child.state[:50]}...")
                score = self.simulate(child)
            else:
                score = self.simulate(node)
            
            print(f"Simulated score: {score:.4f}")
            self.backpropagate(node, score)
            print("Backpropagated score")

        best_child = max(root.children, key=lambda c: c.visits)
        
        mermaid_output = self.generate_mermaid(root, best_child)
        self.save_mermaid_markdown(mermaid_output, root_state, best_child.state, best_child.id)
        
        return best_child.state

    def expand(self, node: Node):
        new_states = self.llm.generate_diverse_answers(node.state, n=3)
        for state in new_states:
            child = Node(state, parent=node, depth=node.depth + 1)
            child.id = self.node_count
            self.node_count += 1
            node.children.append(child)

    def generate_mermaid(self, root: Node, best_node: Node) -> str:
        mermaid = ["graph TD"]
        visited = set()
        best_path = self.get_path_to_best(best_node)

        def add_node(node: Node, is_in_best_path: bool):
            if node.id in visited:
                return
            visited.add(node.id)
            
            if node == best_node:
                style = "style fill:#90EE90 stroke:#006400 stroke-width:4px"
            elif is_in_best_path:
                style = "style fill:#E6F3FF stroke:#4169E1 stroke-width:2px"
            else:
                style = "style fill:#f9f stroke:#333 stroke-width:2px"
            
            label = f"{node.id}[\"Node {node.id}<br>Depth: {node.depth}<br>{node.state[:20]}...<br>Visits: {node.visits}<br>Value: {node.value:.2f}\"]"
            mermaid.append(label)
            mermaid.append(f"{style} {node.id}")
            
            for child in node.children:
                edge_style = "|stroke:#4169E1 stroke-width:2px|" if child in best_path else ""
                mermaid.append(f"{node.id} -->{edge_style} {child.id}")
                add_node(child, child in best_path)

        add_node(root, root in best_path)
        return "\n".join(mermaid)
    
    def get_path_to_best(self, best_node: Node) -> List[Node]:
        path = []
        current = best_node
        while current:
            path.append(current)
            current = current.parent
        return list(reversed(path))

    def save_mermaid_markdown(self, mermaid_content: str, root_state: str, best_answer: str, best_node_id: int):
        filename = "mcts_tree_visualization.md"
        with open(filename, "w") as f:
            f.write("# MCTS Tree Visualization\n\n")
            f.write(f"Initial question: {root_state}\n\n")
            f.write("```mermaid\n")
            f.write(mermaid_content)
            f.write("\n```\n\n")
            f.write(f"## Best Answer (Node {best_node_id})\n\n")
            f.write(f"{best_answer}\n")
        print(f"\nMCTS Tree visualization saved to {filename}")

    def select(self, node: Node) -> Node:
        while node.children:
            if random.random() < self.exploration_factor:
                return random.choice(node.children)
            node = max(node.children, key=lambda c: self.ucb1(c))
        return node

    def ucb1(self, node: Node) -> float:
        if node.visits == 0:
            return float('inf')
        return (node.value / node.visits) + self.C * math.sqrt(math.log(node.parent.visits) / node.visits)

    def simulate(self, node: Node) -> float:
        return self.llm.evaluate_answer(node.state)

    def backpropagate(self, node: Node, score: float):
        while node:
            node.visits += 1
            node.value += score
            node = node.parent

    def save_checkpoint(self, root: Node, filename: str):
        with open(filename, 'w') as f:
            json.dump(root.to_dict(), f)

    @classmethod
    def load_checkpoint(cls, filename: str, llm) -> 'MCTS':
        with open(filename, 'r') as f:
            data = json.load(f)
        mcts = cls(llm)
        mcts.root = mcts.dict_to_node(data)
        return mcts

    def dict_to_node(self, data: Dict) -> Node:
        node = Node(data['state'])
        node.visits = data['visits']
        node.value = data['value']
        for child_data in data['children']:
            child = self.dict_to_node(child_data)
            child.parent = node
            node.children.append(child)
        return node

class LLM:
    def __init__(self, model='gemma2'):
        self.model = model
        self.client = instructor.from_openai(
            OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",  # required, but unused
            ),
            mode=instructor.Mode.JSON,
        )
        self.executor = ThreadPoolExecutor(max_workers=3)

    def generate_diverse_answers(self, prompt: str, n: int) -> List[str]:
        print(f"\nGenerating {n} diverse answers...")
        futures = []
        for i in range(n):
            futures.append(self.executor.submit(self._generate_single_answer, prompt, i))
        
        answers = []
        for future in as_completed(futures):
            answers.append(future.result())
        
        for i, answer in enumerate(answers):
            print(f"Answer {i+1}: {answer[:100]}...")
        
        return answers

    def _generate_single_answer(self, prompt: str, index: int) -> str:
        diversity_prompt = f"Provide a unique and diverse answer to the following question. Focus on aspect {index + 1} of the problem: {prompt}"
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": diversity_prompt,
                    }
                ],
                response_model=str,
            )
            return response
        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"Error: Failed to generate answer {index + 1}"

    def evaluate_answer(self, answer: str) -> float:
        print("\nEvaluating answer...")
        evaluation_prompt = f"Evaluate the following answer based on relevance, accuracy, depth, clarity, and originality. Provide scores from 0 to 1 for each criterion and an overall score.\n\nAnswer: {answer}"
        try:
            evaluation = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": evaluation_prompt,
                    }
                ],
                response_model=EvaluationCriteria,
            )
            print(f"Evaluation results: {evaluation}")
            return evaluation.overall_score
        except Exception as e:
            print(f"Error evaluating answer: {e}")
            return 0.5  # Default to neutral score if evaluation fails

def main():
    llm = LLM()
    mcts = MCTS(llm, C=1.5, max_iterations=20, exploration_factor=0.15, max_depth=2)
    initial_state = "What are the key factors contributing to climate change, and what are the most effective strategies to mitigate its impact?"
    best_answer = mcts.search(initial_state)
    print(f"\nBest answer found:\n{best_answer}")

if __name__ == "__main__":
    main()

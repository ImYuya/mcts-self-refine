from manim import *
import numpy as np

class MCTSNode:
    def __init__(self, label, position):
        self.label = label
        self.position = position
        self.visits = 0
        self.total_value = 0
        self.circle = Circle(radius=0.5, color=WHITE, fill_opacity=0.2)
        self.label_text = Text(f"Node {label}", font_size=20).next_to(self.circle, UP, buff=0.1)
        self.stats_text = Text("V: 0\nA: 0.00", font_size=16).move_to(self.circle.get_center())
        self.mobject = VGroup(self.circle, self.label_text, self.stats_text)
        self.mobject.move_to(position)
        self.children = []

    def update_stats(self):
        avg_value = self.total_value / self.visits if self.visits > 0 else 0
        new_stats = Text(f"V: {self.visits}\nA: {avg_value:.2f}", font_size=16).move_to(self.circle.get_center())
        return new_stats

class MCTSVisualization(Scene):
    def construct(self):
        # Set up the initial tree
        root = MCTSNode("0", (0, 3, 0))
        self.play(Create(root.mobject))
        self.wait()

        # Add child nodes (first level)
        first_level_children = [MCTSNode(str(i), ((i-2)*4, 0, 0)) for i in range(1, 4)]
        root.children = first_level_children
        first_level_arrows = VGroup(*[Arrow(root.circle.get_bottom(), child.circle.get_top(), buff=0.1, color=GRAY) for child in first_level_children])
        self.play(Create(VGroup(*[child.mobject for child in first_level_children])), Create(first_level_arrows))
        self.wait()

        # Add second level children and arrows
        second_level_children = []
        second_level_arrows = VGroup()
        for i, first_child in enumerate(first_level_children):
            children = [MCTSNode(f"{4+i*2+j}", (first_child.position[0] + (j-0.5)*2, -3, 0)) for j in range(2)]
            first_child.children = children
            second_level_children.extend(children)
            arrows = VGroup(*[Arrow(first_child.circle.get_bottom(), child.circle.get_top(), buff=0.1, color=GRAY) for child in children])
            second_level_arrows.add(arrows)
        self.play(Create(VGroup(*[child.mobject for child in second_level_children])), Create(second_level_arrows))
        self.wait()

        # Simulate MCTS iterations
        all_nodes = [root] + first_level_children + second_level_children
        all_arrows = VGroup(first_level_arrows, second_level_arrows)
        for _ in range(30):  # Increased number of iterations for better exploration
            path, arrows = self.select_path(root)
            for i, node in enumerate(path):
                if i > 0:
                    self.play(arrows[i-1].animate.set_color(YELLOW))
                self.play(node.circle.animate.set_color(YELLOW))
                self.wait(0.2)

            value = self.simulate_and_backpropagate(path)
            self.update_node_stats(path, value)
            
            self.play(*[arrow.animate.set_color(GRAY) for arrow in arrows], 
                      *[node.circle.animate.set_color(WHITE) for node in path])

        # Show final selection
        best_node = max(first_level_children, key=lambda x: x.visits)
        self.play(best_node.circle.animate.set_color(GREEN))
        best_label = Text("Best Node", font_size=24, color=GREEN).next_to(best_node.mobject, DOWN)
        self.play(Write(best_label))
        self.wait(2)

        # Remove arrows
        self.play(FadeOut(all_arrows), FadeOut(best_label))
        self.wait(1)

        # Fade out nodes
        self.play(*[FadeOut(node.mobject) for node in all_nodes])
        self.wait(1)

        # Conclusion
        conclusion = Text("MCTS Algorithm Completed", font_size=36)
        self.play(Write(conclusion))
        self.wait(2)

    def select_path(self, root):
        path = [root]
        arrows = VGroup()

        while path[-1].children:
            next_node = self.select_best_child(path[-1])
            if next_node:
                arrows.add(Arrow(path[-1].circle.get_bottom(), next_node.circle.get_top(), buff=0.1, color=YELLOW))
                path.append(next_node)
            else:
                break

        return path, arrows

    def select_best_child(self, parent):
        if not parent.children:
            return None
        
        C = 1.41  # Exploration parameter
        
        def ucb(child):
            if child.visits == 0:
                return float('inf')
            exploitation = child.total_value / child.visits
            exploration = C * np.sqrt(np.log(parent.visits + 1) / (child.visits + 1e-6))
            return exploitation + exploration
        
        return max(parent.children, key=ucb)

    def simulate_and_backpropagate(self, path):
        value = np.random.rand()  # Simulate a random value
        color = interpolate_color(BLUE, RED, value)
        self.play(*[node.circle.animate.set_stroke(color=color) for node in path])
        self.wait(0.2)
        return value

    def update_node_stats(self, path, value):
        for node in path:
            node.visits += 1
            node.total_value += value
            new_stats = node.update_stats()
            self.play(Transform(node.stats_text, new_stats))

# To render the animation, run:
# manim -pql ./src/mcts_manim_visualization.py MCTSVisualization
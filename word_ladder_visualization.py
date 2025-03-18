import networkx as nx
import matplotlib.pyplot as plt
import time
from collections import deque

# Fix the __init__ method in the imported class
class WordLadderSolver:
    def __init__(self, word_list_file=None):
        """Initialize the Word Ladder Solver."""
        # If no word list is provided, use a small built-in dictionary
        if word_list_file:
            with open(word_list_file, 'r') as f:
                self.dictionary = set(word.strip().lower() for word in f)
        else:
            # Small built-in dictionary for demonstration
            self.dictionary = {
                "cold", "cord", "card", "ward", "warm", "worm", "word", "wood", 
                "good", "gold", "golf", "wolf", "wool", "cool", "pool", "poll", 
                "pole", "pope", "pops", "tops", "taps", "tape", "tale", "pale",
                "palm", "calm", "camp", "lamp", "ramp", "raid", "paid", "pain",
                "rain", "main", "mail", "tail", "sail", "soil", "soul", "soup",
                "loop", "look", "book", "boot", "boat", "coat", "coal", "foal",
                "foam", "roam", "road", "toad", "load", "goad", "goal", "goat",
                "boat", "beat", "meat", "meal", "deal", "dear", "bear", "beer",
                "peer", "peel", "feel", "fell", "tell", "tall", "ball", "call",
                "cell", "sell", "seal", "real", "teal", "team", "beam", "beak",
                "peak", "pear", "fear", "tear", "tsar", "star", "stir", "stir",
                "sail", "fail", "fall", "fill", "file", "fire", "hire", "hide",
                "wide", "wine", "fine", "find", "bind", "kind", "mind", "mint",
                "hint", "hunt", "hurt", "hard", "hand", "land", "lend"
            }
        
        # Filter words to ensure consistency in word length groups
        self.words_by_length = {}
        for word in self.dictionary:
            length = len(word)
            if length not in self.words_by_length:
                self.words_by_length[length] = set()
            self.words_by_length[length].add(word)
        
        print(f"Dictionary loaded with {len(self.dictionary)} words")
        print(f"Word length distribution: {[(k, len(v)) for k, v in self.words_by_length.items()]}")
    
    def get_neighbors(self, word):
        """Find all words that differ by exactly one letter from the given word."""
        neighbors = []
        # For each position in the word
        for i in range(len(word)):
            # Try replacing with each letter in the alphabet
            for char in 'abcdefghijklmnopqrstuvwxyz':
                # Skip if it's the same letter
                if char == word[i]:
                    continue
                
                # Create a new word with one letter changed
                new_word = word[:i] + char + word[i+1:]
                
                # If the new word is in our dictionary, it's a valid neighbor
                if new_word in self.words_by_length.get(len(word), set()):
                    neighbors.append(new_word)
        
        return neighbors

def visualize_word_ladder_step_by_step(start_word="tail", end_word="hand", max_nodes=250):
    """
    Create a step-by-step visualization of the word ladder graph building process.
    Shows the full graph first, then animates the BFS exploration.
    
    Color Legend:
    - Green: Start word
    - Red: End word
    - Yellow: Current word being processed
    - Orange: Newly discovered words
    - Purple: Words in queue waiting to be processed
    - Blue: Edges explored in BFS tree
    - Sky Blue: Previously visited nodes
    - Light Gray: Unvisited nodes
    - Red edges: Final solution path
    """
    # Create a solver instance
    solver = WordLadderSolver()
    
    # Validate words
    if len(start_word) != len(end_word):
        print(f"Error: Words must be same length. Got '{start_word}' and '{end_word}'")
        return None, None
        
    if start_word not in solver.dictionary:
        print(f"Error: '{start_word}' not in dictionary")
        return None, None
        
    if end_word not in solver.dictionary:
        print(f"Error: '{end_word}' not in dictionary")
        return None, None
    
    # Build the complete graph first
    print(f"Building complete word graph for words of length {len(start_word)}...")
    G = nx.Graph()
    
    # Add all words of the same length
    word_length = len(start_word)
    all_words = list(solver.words_by_length.get(word_length, set()))
    if len(all_words) > max_nodes:
        print(f"Too many words ({len(all_words)}), limiting to {max_nodes}")
        all_words = all_words[:max_nodes]
    
    # Add all nodes to the graph
    for word in all_words:
        G.add_node(word)
    
    # Add all edges
    edge_count = 0
    for i, word in enumerate(all_words):
        for neighbor in solver.get_neighbors(word):
            if neighbor in all_words:
                G.add_edge(word, neighbor)
                edge_count += 1
    
    print(f"Graph built with {len(G.nodes)} nodes and {edge_count} edges")
    
    # Calculate shortest path
    try:
        shortest_path = nx.shortest_path(G, start_word, end_word)
        print(f"Shortest path found: {' → '.join(shortest_path)}")
    except nx.NetworkXNoPath:
        shortest_path = None
        print(f"No path exists between '{start_word}' and '{end_word}'")
    
    # Calculate a better layout once and reuse it
    pos = nx.spring_layout(G, seed=42)
    
    # Create the figure
    plt.figure(figsize=(14, 12))
    
    # First, draw the entire graph
    plt.clf()
    plt.title(f"Word Ladder Graph: {start_word.upper()} to {end_word.upper()}", fontsize=16)
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5)
    nx.draw_networkx_nodes(G, pos, node_size=80, node_color='lightgray', alpha=0.7)
    
    # Highlight start and end nodes
    nx.draw_networkx_nodes(G, pos, nodelist=[start_word], node_size=300, node_color='green', alpha=1.0)
    nx.draw_networkx_nodes(G, pos, nodelist=[end_word], node_size=300, node_color='red', alpha=1.0)
    
    # Add labels for start and end
    nx.draw_networkx_labels(G, pos, labels={start_word: start_word, end_word: end_word}, 
                           font_size=12, font_weight='bold')
    
    plt.text(0.05, 0.95, "Complete Word Graph", transform=plt.gca().transAxes, 
             fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.axis('off')
    plt.tight_layout()
    plt.pause(3)  # Pause to show the full graph
    
    # Now simulate the BFS exploration
    queue = deque([(start_word, [start_word])])
    visited = {start_word}
    queue_words = {start_word}  # Track words in queue
    
    step = 0
    found_path = None
    visited_order = [start_word]  # Track order of visited nodes
    
    # For tracking parent relationships in the BFS tree
    parents = {start_word: None}
    
    # Create color legend
    def add_color_legend(ax):
        legend_text = [
            "Color Legend:",
            "⬤ Green: Start word",
            "⬤ Red: End word",
            "⬤ Yellow: Current word being processed",
            "⬤ Orange: Newly discovered words",
            "⬤ Purple: Words in queue waiting to be processed",
            "⬤ Sky Blue: Previously visited nodes",
            "⬤ Light Gray: Unvisited nodes",
            "— Blue: Explored edges",
            "— Orange: Newly discovered edges",
            "— Red: Solution path"
        ]
        
        legend_colors = [
            "black",
            "green",
            "red",
            "yellow",
            "orange",
            "purple",
            "skyblue",
            "lightgray",
            "blue",
            "orange",
            "red"
        ]
        
        y_pos = 0.05
        for i, (text, color) in enumerate(zip(legend_text, legend_colors)):
            if i == 0:  # Title
                ax.text(0.75, y_pos + 0.25, text, transform=ax.transAxes, 
                      fontsize=10, fontweight='bold', 
                      bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
            else:
                ax.text(0.75, y_pos + 0.25 - i*0.02, text, transform=ax.transAxes, 
                      fontsize=9, color=color,
                      bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1'))
    
    while queue and len(visited) < max_nodes:
        step += 1
        current_word, path = queue.popleft()
        
        # Check if we found the target
        if current_word == end_word:
            found_path = path
            break
        
        # Process all neighbors before redrawing
        new_neighbors = []
        for neighbor in solver.get_neighbors(current_word):
            if neighbor not in visited:
                visited.add(neighbor)
                visited_order.append(neighbor)
                parents[neighbor] = current_word
                queue.append((neighbor, path + [neighbor]))
                queue_words.add(neighbor)
                new_neighbors.append(neighbor)
        
        # Skip redrawing if no new neighbors
        if not new_neighbors:
            continue
        
        # Redraw the graph with updated colors
        plt.clf()
        plt.title(f"Word Ladder BFS Exploration: Step {step}", fontsize=16)
        
        # Draw all edges lightly
        nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5)
        
        # Draw explored edges
        explored_edges = [(parents[word], word) for word in visited if parents[word] is not None]
        nx.draw_networkx_edges(G, pos, edgelist=explored_edges, alpha=0.8, width=1.5, edge_color='blue')
        
        # Draw the newly discovered edges
        new_edges = [(current_word, neighbor) for neighbor in new_neighbors]
        nx.draw_networkx_edges(G, pos, edgelist=new_edges, alpha=1.0, width=2, edge_color='orange')
        
        # Draw all nodes with different colors based on status
        unvisited_nodes = [n for n in G.nodes if n not in visited]
        visited_nodes = list(visited - {start_word, end_word} - set(new_neighbors) - {current_word})
        
        # Get words in queue (excluding the ones we just processed)
        queue_nodes = [word for word, _ in queue if word != current_word]
        queue_words.remove(current_word)  # Remove current word from queue tracking
        for neighbor in new_neighbors:
            queue_words.add(neighbor)
            
        # Remove overlap between different node groups
        queue_only = [n for n in queue_words if n not in new_neighbors and n != current_word 
                     and n != start_word and n != end_word]
        
        # Draw unvisited nodes
        nx.draw_networkx_nodes(G, pos, nodelist=unvisited_nodes, 
                              node_size=80, node_color='lightgray', alpha=0.5)
        
        # Draw visited nodes
        if visited_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=visited_nodes, 
                                  node_size=100, node_color='skyblue', alpha=0.8)
        
        # Draw queue nodes
        if queue_only:
            nx.draw_networkx_nodes(G, pos, nodelist=queue_only, 
                                  node_size=120, node_color='purple', alpha=0.8)
        
        # Draw newly visited nodes
        nx.draw_networkx_nodes(G, pos, nodelist=new_neighbors, 
                              node_size=150, node_color='orange', alpha=1.0)
        
        # Draw current node
        nx.draw_networkx_nodes(G, pos, nodelist=[current_word], 
                              node_size=200, node_color='yellow', alpha=1.0)
        
        # Always highlight start and end nodes
        nx.draw_networkx_nodes(G, pos, nodelist=[start_word], 
                              node_size=300, node_color='green', alpha=1.0)
        nx.draw_networkx_nodes(G, pos, nodelist=[end_word], 
                              node_size=300, node_color='red', alpha=1.0)
        
        # Add labels
        important_nodes = [start_word, end_word, current_word] + new_neighbors
        labels = {node: node for node in important_nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold')
        
        # Add status text
        status_text = f"Words explored: {len(visited)} of {len(G.nodes)}"
        plt.text(0.05, 0.95, status_text, transform=plt.gca().transAxes, 
                fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
                
        # Add color legend
        add_color_legend(plt.gca())
        
        plt.axis('off')
        plt.tight_layout()
        plt.pause(2.5)  # Slow down animation
    
    # Show the final path if found
    if found_path:
        plt.clf()
        plt.title(f"Word Ladder Solution: {start_word} → {end_word}", fontsize=16)
        
        # Draw all edges lightly
        nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5)
        
        # Draw the BFS tree
        explored_edges = [(parents[word], word) for word in visited if parents[word] is not None]
        nx.draw_networkx_edges(G, pos, edgelist=explored_edges, alpha=0.4, width=1, edge_color='blue')
        
        # Draw the solution path edges
        path_edges = [(found_path[i], found_path[i+1]) for i in range(len(found_path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, alpha=1.0, width=3, edge_color='red')
        
        # Draw all nodes
        nx.draw_networkx_nodes(G, pos, node_size=80, node_color='lightgray', alpha=0.4)
        
        # Draw visited nodes
        visited_nodes = list(visited - set(found_path) - {start_word, end_word})
        nx.draw_networkx_nodes(G, pos, nodelist=visited_nodes, 
                              node_size=100, node_color='skyblue', alpha=0.6)
        
        # Draw path nodes
        nx.draw_networkx_nodes(G, pos, nodelist=found_path, 
                              node_size=200, node_color='yellow', alpha=1.0)
        
        # Highlight start and end nodes
        nx.draw_networkx_nodes(G, pos, nodelist=[start_word], 
                              node_size=300, node_color='green', alpha=1.0)
        nx.draw_networkx_nodes(G, pos, nodelist=[end_word], 
                              node_size=300, node_color='red', alpha=1.0)
        
        # Add labels for path nodes
        labels = {node: node for node in found_path}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=12, font_weight='bold')
        
        # Add solution info
        solution_text = f"Solution found in {len(found_path)-1} steps: {' → '.join(found_path)}"
        plt.figtext(0.5, 0.01, solution_text, ha='center', fontsize=14, 
                   bbox=dict(facecolor='lightyellow', alpha=0.9, boxstyle='round,pad=0.5'))
        
        plt.axis('off')
        plt.tight_layout()
    
    plt.show()
    return G, found_path

# Example usage
if __name__ == "__main__":
    print("Word Ladder Visualization Demo")
    print("==============================")
    print("Recommended pairs:")
    print("1. tail → hand")
    print("2. cold → warm")
    print("3. wood → fire")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        start, end = "tail", "hand"
    elif choice == "2":
        start, end = "cold", "warm"
    elif choice == "3":
        start, end = "wood", "fire"
    else:
        try:
            start, end = choice.split()
        except:
            print("Invalid input. Using default: tail → hand")
            start, end = "tail", "hand"
    
    print(f"\nVisualizing word ladder from {start} to {end}...")
    G, ladder = visualize_word_ladder_step_by_step(start, end)
    
    if ladder:
        print(f"Found ladder: {' → '.join(ladder)}")
    else:
        print("No ladder found")
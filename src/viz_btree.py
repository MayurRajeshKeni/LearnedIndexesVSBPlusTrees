"""
Visualization-Oriented B+ Tree (Teaching Implementation)

Used specifically for the UI to illustrate internal tree traversal, node splitting,
and search path tracing. Standard production libraries (like BTrees.OOBTree) cannot
expose pointer traversals or internal node state.

Properties:
- Configurable order B (default B=4, meaning max 3 keys and 4 children per node).
- Explicit leaf node horizontal linking (next_leaf) for sequential scans.
- lookup_trace(key): detailed trajectory of nodes and levels visited.
- insert(key, value): supports dynamic insertion with node splits.
- to_dot(highlight_path): exports valid Graphviz DOT string with path highlighting.
"""

from typing import Optional, List, Dict, Any, Tuple
import bisect


class BTreeNode:
    _id_counter = 0

    def __init__(self, is_leaf: bool = False):
        BTreeNode._id_counter += 1
        self.id = BTreeNode._id_counter
        self.is_leaf = is_leaf
        self.keys: List[int] = []

    def __repr__(self):
        return f"<Node {self.id} leaf={self.is_leaf} keys={self.keys}>"


class InternalNode(BTreeNode):
    def __init__(self):
        super().__init__(is_leaf=False)
        self.children: List[BTreeNode] = []


class LeafNode(BTreeNode):
    def __init__(self):
        super().__init__(is_leaf=True)
        self.values: List[Any] = []
        self.next_leaf: Optional["LeafNode"] = None


class VizBTree:
    def __init__(self, order: int = 4):
        """
        :param order: Max number of children for internal nodes (default 4).
        """
        if order < 3:
            raise ValueError("B+ Tree order must be at least 3")
        self.order = order
        self.root: Optional[BTreeNode] = LeafNode()
        self.split_events: List[Dict[str, Any]] = []

    def clear(self):
        BTreeNode._id_counter = 0
        self.root = LeafNode()
        self.split_events.clear()

    def build(self, sorted_keys: List[int], values: Optional[List[Any]] = None):
        """Build tree by inserting keys in order."""
        self.clear()
        if values is None:
            values = list(range(len(sorted_keys)))

        for k, v in zip(sorted_keys, values):
            self.insert(int(k), v)

    def lookup(self, key: int) -> Optional[Any]:
        """Standard B+ Tree search returning value or None."""
        trace = self.lookup_trace(key)
        return trace["value"] if trace["found"] else None

    def lookup_trace(self, key: int) -> Dict[str, Any]:
        """
        Traverse B+ Tree while recording full path:
        - visited_nodes: list of nodes inspected
        - levels_visited: depth of traversal
        - comparisons: key comparisons made
        """
        key = int(key)
        visited = []
        curr = self.root
        level = 0
        comparisons = 0

        while not curr.is_leaf:
            # Binary search or scan within internal node
            # Child i is followed: child 0 if key < keys[0], etc.
            keys = curr.keys
            child_idx = 0
            while child_idx < len(keys) and key >= keys[child_idx]:
                comparisons += 1
                child_idx += 1
            if child_idx < len(keys):
                comparisons += 1

            visited.append({
                "node_id": curr.id,
                "level": level,
                "is_leaf": False,
                "keys": list(keys),
                "child_index": child_idx,
            })

            curr = curr.children[child_idx]
            level += 1

        # Now in leaf node
        keys = curr.keys
        found = False
        val = None
        match_idx = None

        idx = bisect.bisect_left(keys, key)
        comparisons += len(keys)
        if idx < len(keys) and keys[idx] == key:
            found = True
            val = curr.values[idx]
            match_idx = idx

        visited.append({
            "node_id": curr.id,
            "level": level,
            "is_leaf": True,
            "keys": list(keys),
            "match_index": match_idx,
        })

        return {
            "key": key,
            "found": found,
            "value": val,
            "visited_nodes": visited,
            "num_visited": len(visited),
            "max_level": level,
            "comparisons": comparisons,
        }

    def insert(self, key: int, value: Any = None):
        """Insert (key, value) with recursive node splitting."""
        key = int(key)
        if value is None:
            value = key

        split_res = self._insert_internal(self.root, key, value)
        if split_res is not None:
            # Root split: create new root
            promoted_key, right_child = split_res
            new_root = InternalNode()
            new_root.keys = [promoted_key]
            new_root.children = [self.root, right_child]
            self.root = new_root
            self.split_events.append({
                "type": "root_split",
                "promoted_key": promoted_key,
                "new_root_id": new_root.id
            })

    def _insert_internal(self, node: BTreeNode, key: int, value: Any) -> Optional[Tuple[int, BTreeNode]]:
        """Recursive helper for insert. Returns (promoted_key, new_right_node) on split."""
        if node.is_leaf:
            idx = bisect.bisect_left(node.keys, key)
            if idx < len(node.keys) and node.keys[idx] == key:
                # Update duplicate
                node.values[idx] = value
                return None

            node.keys.insert(idx, key)
            node.values.insert(idx, value)

            # Check if leaf needs split (order - 1 is capacity)
            if len(node.keys) >= self.order:
                return self._split_leaf(node)
            return None
        else:
            # Internal node: find child
            idx = 0
            while idx < len(node.keys) and key >= node.keys[idx]:
                idx += 1

            child = node.children[idx]
            split_res = self._insert_internal(child, key, value)

            if split_res is not None:
                promoted_key, right_child = split_res
                insert_pos = bisect.bisect_right(node.keys, promoted_key)
                node.keys.insert(insert_pos, promoted_key)
                node.children.insert(insert_pos + 1, right_child)

                if len(node.keys) >= self.order:
                    return self._split_internal(node)

            return None

    def _split_leaf(self, leaf: LeafNode) -> Tuple[int, LeafNode]:
        mid = len(leaf.keys) // 2
        right = LeafNode()
        right.keys = leaf.keys[mid:]
        right.values = leaf.values[mid:]

        leaf.keys = leaf.keys[:mid]
        leaf.values = leaf.values[:mid]

        right.next_leaf = leaf.next_leaf
        leaf.next_leaf = right

        promoted_key = right.keys[0]
        self.split_events.append({
            "type": "leaf_split",
            "promoted_key": promoted_key,
            "left_id": leaf.id,
            "right_id": right.id
        })
        return promoted_key, right

    def _split_internal(self, node: InternalNode) -> Tuple[int, InternalNode]:
        mid = len(node.keys) // 2
        promoted_key = node.keys[mid]

        right = InternalNode()
        right.keys = node.keys[mid + 1:]
        right.children = node.children[mid + 1:]

        node.keys = node.keys[:mid]
        node.children = node.children[:mid + 1]

        self.split_events.append({
            "type": "internal_split",
            "promoted_key": promoted_key,
            "left_id": node.id,
            "right_id": right.id
        })
        return promoted_key, right

    def to_dot(self, highlight_path: Optional[List[int]] = None, max_nodes: int = 50) -> str:
        """
        Generate Graphviz DOT representation.
        Nodes on highlight_path are visually accented.
        Leaves are linked horizontally via dashed next pointers.
        """
        highlight_set = set(highlight_path) if highlight_path else set()

        lines = [
            'digraph BPlusTree {',
            '    rankdir=TB;',
            '    node [shape=record, fontname="Helvetica", fontsize=10, height=0.35];',
            '    edge [fontname="Helvetica", fontsize=9, color="#555555"];',
            '    graph [splines=true, bgcolor="transparent"];'
        ]

        if not self.root:
            lines.append('    empty [label="Empty Tree"];')
            lines.append('}')
            return '\n'.join(lines)

        # BFS collection
        all_nodes: List[BTreeNode] = []
        queue = [self.root]
        while queue:
            curr = queue.pop(0)
            all_nodes.append(curr)
            if not curr.is_leaf:
                for ch in curr.children:
                    queue.append(ch)

        # If tree is very large, optionally prune non-highlighted nodes
        display_nodes = all_nodes
        if len(all_nodes) > max_nodes and highlight_set:
            # Keep root, direct ancestors, highlighted nodes, and their siblings
            display_nodes = [n for n in all_nodes if n.id in highlight_set or n == self.root][:max_nodes]

        rendered_ids = {n.id for n in display_nodes}

        # Render Nodes
        leaves: List[LeafNode] = []
        for node in display_nodes:
            is_highlighted = node.id in highlight_set
            bg_color = "#ffeaa7" if is_highlighted else ("#edf2f7" if node.is_leaf else "#e2e8f0")
            border_color = "#d63031" if is_highlighted else ("#3182ce" if not node.is_leaf else "#4a5568")
            pen_width = "2.5" if is_highlighted else "1.0"

            # Node label ports
            if node.is_leaf:
                leaves.append(node)
                key_strs = [f"<f{i}> {k}" for i, k in enumerate(node.keys)]
                label = " | ".join(key_strs) if key_strs else "empty"
                label = f"{{ {label} }}"
            else:
                parts = []
                for i in range(len(node.keys)):
                    parts.append(f"<p{i}> &bull;")
                    parts.append(f"<f{i}> {node.keys[i]}")
                parts.append(f"<p{len(node.keys)}> &bull;")
                label = " | ".join(parts)

            lines.append(
                f'    node_{node.id} [label="{label}", style="filled", fillcolor="{bg_color}", '
                f'color="{border_color}", penwidth={pen_width}];'
            )

        # Render Parent -> Child Edges
        for node in display_nodes:
            if not node.is_leaf:
                for idx, ch in enumerate(node.children):
                    if ch.id in rendered_ids:
                        is_edge_highlight = (node.id in highlight_set and ch.id in highlight_set)
                        edge_color = "#d63031" if is_edge_highlight else "#718096"
                        pen_w = "2.2" if is_edge_highlight else "1.0"
                        lines.append(
                            f'    node_{node.id}:p{idx} -> node_{ch.id} [color="{edge_color}", penwidth={pen_w}];'
                        )

        # Render Leaf Horizontal Chain (next_leaf)
        # Group leaves in same rank
        if leaves:
            leaf_ids_in_graph = [f"node_{lf.id}" for lf in leaves if lf.id in rendered_ids]
            if len(leaf_ids_in_graph) > 1:
                lines.append(f'    {{ rank=same; {" ".join(leaf_ids_in_graph)} }}')

            for lf in leaves:
                if lf.next_leaf and lf.next_leaf.id in rendered_ids:
                    lines.append(
                        f'    node_{lf.id} -> node_{lf.next_leaf.id} '
                        f'[style=dashed, constraint=false, color="#38a169", arrowhead=vee, penwidth=1.2];'
                    )

        lines.append('}')
        return '\n'.join(lines)

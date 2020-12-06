import heapq
import argparse

# -------
# utils
# -------
def memoize(fn, slot=None, maxsize=32):
    """Memoize fn: make it remember the computed value for any argument list.
    If slot is specified, store result in that slot of first argument.
    If slot is false, use lru_cache for caching the values."""
    if slot:
        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                return val
    else:
        @functools.lru_cache(maxsize=maxsize)
        def memoized_fn(*args):
            return fn(*args)

    return memoized_fn

def manhattan(node):
    state = node.state
    index_goal = {0:[2,2], 1:[0,0], 2:[0,1], 3:[0,2], 4:[1,0], 5:[1,1], 6:[1,2], 7:[2,0], 8:[2,1]}
    index_state = {}
    index = [[0,0], [0,1], [0,2], [1,0], [1,1], [1,2], [2,0], [2,1], [2,2]]
    x, y = 0, 0

    for i in range(len(state)):
        index_state[state[i]] = index[i]

    mhd = 0

    for i in range(8):
        for j in range(2):
            mhd = abs(index_goal[i][j] - index_state[i][j]) + mhd

class PriorityQueue:
    """A Queue in which the minimum (or maximum) element (as determined by f and
    order) is returned first.
    If order is 'min', the item with minimum f(x) is
    returned first; if order is 'max', then it is the item with maximum f(x).
    Also supports dict-like lookup."""

    def __init__(self, order='min', f=lambda x: x):
        self.heap = []
        if order == 'min':
            self.f = f
        elif order == 'max':  # now item with max f(x)
            self.f = lambda x: -f(x)  # will be popped first
        else:
            raise ValueError("Order must be either 'min' or 'max'.")

    def append(self, item):
        """Insert item at its correct position."""
        heapq.heappush(self.heap, (self.f(item), item))

    def extend(self, items):
        """Insert each item in items at its correct position."""
        for item in items:
            self.append(item)

    def pop(self):
        """Pop and return the item (with min or max f(x) value)
        depending on the order."""
        if self.heap:
            return heapq.heappop(self.heap)[1]
        else:
            raise Exception('Trying to pop from empty PriorityQueue.')

    def __len__(self):
        """Return current capacity of PriorityQueue."""
        return len(self.heap)

    def __contains__(self, key):
        """Return True if the key is in PriorityQueue."""
        return any([item == key for _, item in self.heap])

    def __getitem__(self, key):
        """Returns the first value associated with key in PriorityQueue.
        Raises KeyError if key is not present."""
        for value, item in self.heap:
            if item == key:
                return value
        raise KeyError(str(key) + " is not in the priority queue")

    def __delitem__(self, key):
        """Delete the first occurrence of key."""
        try:
            del self.heap[[item == key for _, item in self.heap].index(True)]
        except ValueError:
            raise KeyError(str(key) + " is not in the priority queue")
        heapq.heapify(self.heap)


class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state. Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node. Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
        return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    def print_path(self):
        for n in self.path():
            print(n.state, end=' ')
        print('\n')

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        # We use the hash value of the state
        # stored in the node instead of the node
        # object itself to quickly search a node
        # with the same state in a Hash Table
        return hash(self.state)

class Graph:
    """A graph connects nodes (vertices) by edges (links). Each edge can also
    have a length associated with it. The constructor call is something like:
        g = Graph({'A': {'B': 1, 'C': 2})
    this makes a graph with 3 nodes, A, B, and C, with an edge of length 1 from
    A to B,  and an edge of length 2 from A to C. You can also do:
        g = Graph({'A': {'B': 1, 'C': 2}, directed=False)
    This makes an undirected graph, so inverse links are also added. The graph
    stays undirected; if you add more links with g.connect('B', 'C', 3), then
    inverse link is also added. You can use g.nodes() to get a list of nodes,
    g.get('A') to get a dict of links out of A, and g.get('A', 'B') to get the
    length of the link from A to B. 'Lengths' can actually be any object at
    all, and nodes can be any hashable object."""

    def __init__(self, graph_dict=None, directed=True):
        self.graph_dict = graph_dict or {}
        self.directed = directed
        if not directed:
            self.make_undirected()

    def make_undirected(self):
        """Make a digraph into an undirected graph by adding symmetric edges."""
        for a in list(self.graph_dict.keys()):
            for (b, dist) in self.graph_dict[a].items():
                self.connect1(b, a, dist)

    def connect(self, A, B, distance=1):
        """Add a link from A and B of given distance, and also add the inverse
        link if the graph is undirected."""
        self.connect1(A, B, distance)
        if not self.directed:
            self.connect1(B, A, distance)

    def connect1(self, A, B, distance):
        """Add a link from A to B of given distance, in one direction only."""
        self.graph_dict.setdefault(A, {})[B] = distance

    def get(self, a, b=None):
        """Return a link distance or a dict of {node: distance} entries.
        .get(a,b) returns the distance or None;
        .get(a) returns a dict of {node: distance} entries, possibly {}."""
        links = self.graph_dict.setdefault(a, {})
        if b is None:
            return links
        else:
            return links.get(b)

    def nodes(self):
        """Return a list of nodes in the graph."""
        s1 = set([k for k in self.graph_dict.keys()])
        s2 = set([k2 for v in self.graph_dict.values() for k2, v2 in v.items()])
        nodes = s1.union(s2)
        return list(nodes)


def UndirectedGraph(graph_dict=None):
    """Build a Graph where every edge (including future ones) goes both ways."""
    return Graph(graph_dict=graph_dict, directed=False)

class Problem:
    """The abstract class for a formal problem. You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value. Hill Climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError


class GraphProblem(Problem):
    """The problem of searching a graph from one node to another."""

    def __init__(self, initial, goal, graph):
        super().__init__(initial, goal)
        self.graph = graph

    def actions(self, A):
        """The actions at a graph node are just its neighbors."""
        return list(self.graph.get(A).keys())

    def result(self, state, action):
        """The result of going to a neighbor is just that neighbor."""
        return action

    def path_cost(self, cost_so_far, A, action, B):
        return cost_so_far + (self.graph.get(A, B) or np.inf)

    def find_min_edge(self):
        """Find minimum value of edges."""
        m = np.inf
        for d in self.graph.graph_dict.values():
            local_min = min(d.values())
            m = min(m, local_min)

        return m

    def h(self, node):
        """h function is straight-line distance from a node's state to goal."""
        # locs = getattr(self.graph, 'locations', None)
        dds = getattr(self.graph, 'direct_distances', None)
        if dds:
            if type(node) is str:
                # print(node, dds[node], getattr(node, 'path_cost', None))
                # return int(distance(locs[node], locs[self.goal]))
                return dds[node][self.goal]

            # return int(distance(locs[node.state], locs[self.goal])) #dds[node.state]
            # print(node, dds[node.state], getattr(node, 'path_cost', None))
            return dds[node.state][self.goal]
        else:
            return np.inf

class EightPuzzle(Problem):
    """ The problem of sliding tiles numbered from 1 to 8 on a 3x3 board, where one of the
    squares is a blank. A state is represented as a tuple of length 9, where  element at
    index i represents the tile number  at index i (0 if it's an empty square) """

    def __init__(self, initial, goal=(1, 2, 3, 4, 5, 6, 7, 8, 0)):
        """ Define goal state and initialize a problem """
        super().__init__(initial, goal)

    def find_blank_square(self, state):
        """Return the index of the blank square in a given state"""

        return state.index(0)

    def actions(self, state):
        """ Return the actions that can be executed in the given state.
        The result would be a list, since there are only four possible actions
        in any given state of the environment """

        possible_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        index_blank_square = self.find_blank_square(state)

        if index_blank_square % 3 == 0:
            possible_actions.remove('LEFT')
        if index_blank_square < 3:
            possible_actions.remove('UP')
        if index_blank_square % 3 == 2:
            possible_actions.remove('RIGHT')
        if index_blank_square > 5:
            possible_actions.remove('DOWN')

        return possible_actions

    def result(self, state, action):
        """ Given state and action, return a new state that is the result of the action.
        Action is assumed to be a valid action in the state """

        # blank is the index of the blank square
        blank = self.find_blank_square(state)
        new_state = list(state)

        delta = {'UP': -3, 'DOWN': 3, 'LEFT': -1, 'RIGHT': 1}
        neighbor = blank + delta[action]
        new_state[blank], new_state[neighbor] = new_state[neighbor], new_state[blank]

        return tuple(new_state)

    def goal_test(self, state):
        """ Given a state, return True if state is a goal state or False, otherwise """

        return state == self.goal

    def check_solvability(self, state):
        """ Checks if the given state is solvable """

        inversion = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if (state[i] > state[j]) and state[i] != 0 and state[j] != 0:
                    inversion += 1

        return inversion % 2 == 0

    def h(self, node):
        """ Return the heuristic value for a given state. Default heuristic function used is
        h(n) = number of misplaced tiles """

        return sum(s != g for (s, g) in zip(node.state, self.goal))

# -------
# search algorithms
# -------
def best_first_graph_search(problem, f, display=False):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    f = memoize(f, 'f')
    node = Node(problem.initial)
    print("Initial node: ", problem.initial, "\n")
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    i=1
    while frontier:
        print("Frontier on start of iteration", i, ": ", frontier.heap)
        node = frontier.pop()
        print("Node being tested: " + str(node.state))
        print(node.path())
        if problem.goal_test(node.state):
            if display:
                print(len(explored), "paths have been expanded and", len(frontier), "paths remain in the frontier")
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
        print("Frontier on end of iteration", i, ": ", frontier.heap, "\n")
        i += 1
    return None


def astar_search(problem, h=None, display=False):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n), display)

# -------
# problems
# -------
romania_map = UndirectedGraph(dict(
    Arad=dict(Zerind=75, Sibiu=140, Timisoara=118),
    Bucharest=dict(Urziceni=85, Pitesti=101, Giurgiu=90, Fagaras=211),
    Craiova=dict(Drobeta=120, Rimnicu=146, Pitesti=138),
    Drobeta=dict(Mehadia=75),
    Eforie=dict(Hirsova=86),
    Fagaras=dict(Sibiu=99),
    Hirsova=dict(Urziceni=98),
    Iasi=dict(Vaslui=92, Neamt=87),
    Lugoj=dict(Timisoara=111, Mehadia=70),
    Oradea=dict(Zerind=71, Sibiu=151),
    Pitesti=dict(Rimnicu=97),
    Rimnicu=dict(Sibiu=80),
    Urziceni=dict(Vaslui=142)))

romania_map.direct_distances = dict(
    Arad=366, Bucharest=0, Craiova=160, Drobeta=242, Eforie=161,
    Fagaras=178, Giurgiu=77, Hirsova=151, Iasi=226, Lugoj=244,
    Mehadia=241, Neamt=234, Oradea=380, Pitesti=98, Rimnicu=193,
    Sibiu=253, Timisoara=329, Urziceni=80, Vaslui=199, Zerind=374
)

# romania_problem = GraphProblem('Arad','Bucharest',romania_map)
# astar_search(romania_problem)


metro_map = UndirectedGraph(dict(
    E1=dict(E2=10.0),
    E2=dict(E3=8.5, E9=10.0, E10=3.5),
    E3=dict(E4=6.3, E9=9.4, E13=18.7),
    E4=dict(E5=13.0, E8=15.3, E13=12.8),
    E5=dict(E6=3.0, E7=2.4, E8=30.0),
    E8=dict(E9=9.6, E12=6.4),
    E9=dict(E11=12.2),
    E13=dict(E14=5.1)
))

print("\nStarting metro problem\n")
metro_map.direct_distances = dict(
    E1  = dict(E1=0,    E2=10,   E3=18.5, E4=24.8, E5=36.4, E6=38.8, E7=35.8, E8=25.4, E9=17.6, E10=9.1,  E11=16.7, E12=27.3, E13=27.6, E14=29.8),
    E2  = dict(E1=10,   E2=0,    E3=8.5,  E4=14.8, E5=26.6, E6=29.1, E7=26.1, E8=17.3, E9=10,   E10=3.5,  E11=15.5, E12=20.9, E13=19.1, E14=21.8),
    E3  = dict(E1=18.5, E2=8.5,  E3=0,    E4=6.3,  E5=18.2, E6=20.6, E7=17.6, E8=13.6, E9=9.4,  E10=10.3, E11=19.5, E12=19.1, E13=12.1, E14=16.6),
    E4  = dict(E1=24.8, E2=14.8, E3=6.3 , E4=0,    E5=12,   E6=14.4, E7=11.5, E8=12.4, E9=12.6, E10=16.7, E11=23.6, E12=18.6, E13=10.6, E14=15.4),
    E5  = dict(E1=36.4, E2=26.6, E3=18.2, E4=12,   E5=0,    E6=3,    E7=2.4,  E8=19.4, E9=23.3, E10=28.2, E11=34.2, E12=24.8, E13=14.5, E14=17.9),
    E6  = dict(E1=38.8, E2=29.1, E3=20.6, E4=14.1, E5=3,    E6=0,    E7=3.3,  E8=22.3, E9=25.7, E10=30.3, E11=36.7, E12=27.6, E13=15.2, E14=18.2),
    E7  = dict(E1=35.8, E2=26.1, E3=17.6, E4=11.5, E5=2.4,  E6=3.3,  E7=0,    E8=20,   E9=23,   E10=27.3, E11=34.2, E12=25.7, E13=12.4, E14=15.6),
    E8  = dict(E1=25.4, E2=17.3, E3=13.6, E4=12.4, E5=19.4, E6=22.3, E7=20,   E8=0,    E9=8.2,  E10=20.3, E11=16.1, E12=6.4,  E13=22.7, E14=27.6),
    E9  = dict(E1=17.6, E2=10,   E3=9.4,  E4=12.6, E5=23.3, E6=25.7, E7=23,   E8=8.2,  E9=0,    E10=13.5, E11=11.2, E12=10.9, E13=21.2, E14=26.6),
    E10 = dict(E1=9.1,  E2=3.5,  E3=10.3, E4=16.7, E5=28.2, E6=30.3, E7=27.3, E8=20.3, E9=13.5, E10=0,    E11=17.6, E12=24.2, E13=18.7, E14=21.2),
    E11 = dict(E1=16.7, E2=15.5, E3=19.5, E4=23.6, E5=34.2, E6=36.7, E7=34.2, E8=16.1, E9=11.2, E10=17.6, E11=0,    E12=14.2, E13=31.5, E14=35.5),
    E12 = dict(E1=27.3, E2=20.9, E3=19.1, E4=18.6, E5=24.8, E6=27.6, E7=25.7, E8=6.4,  E9=10.9, E10=24.2, E11=14.2, E12=0,    E13=28.8, E14=33.6),
    E13 = dict(E1=27.6, E2=19.1, E3=12.1, E4=10.6, E5=14.5, E6=15.2, E7=12.4, E8=22.7, E9=21.2, E10=18.7, E11=31.5, E12=28.8, E13=0,    E14=5.1),
    E14 = dict(E1=29.8, E2=21.8, E3=16.6, E4=15.4, E5=17.9, E6=18.2, E7=15.6, E8=27.6, E9=26.6, E10=21.2, E11=35.5, E12=33.6, E13=5.1,  E14=0),
)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--initial", required=True, type=str,
                help="Initial node")
ap.add_argument("-g", "--goal", required=True, type=str,
                help="Goal node")
args = vars(ap.parse_args())

metro_problem = GraphProblem(args["initial"],args["goal"],metro_map)
search = astar_search(metro_problem)
print("Path from initial node:", search.solution(), "\n")


# print("\nStarting 8-puzzle problem\n")
# puzzle = EightPuzzle((1,8,2,0,4,3,7,6,5))
# print(puzzle.check_solvability((1,8,2,0,4,3,7,6,5)))
# astar_search(puzzle)
# print(astar_search(puzzle).solution())

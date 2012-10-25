from abc import ABCMeta, abstractmethod
import pydot
from collections import namedtuple, defaultdict

Cell = namedtuple('Cell', ['row', 'column'])

class Maze:
    __metaclass__ = ABCMeta

    def __init__(self, num_rows, num_columns, start_tuple, goal_tuple, wall_tuples):
        self.num_rows = num_rows
        self.num_columns = num_columns

        self.start = Cell(*start_tuple)
        self.goal = Cell(*goal_tuple)

        self.walls = frozenset([frozenset([Cell(*cell) for cell in wall_tuple]) 
            for wall_tuple in wall_tuples])


    def cell_valid(self, cell):
        return cell.row >= 0 and cell.row < self.num_rows \
                and cell.column >= 0 and cell.column < self.num_columns

    def blocked(self, from_cell, to_cell):
        return frozenset([from_cell, to_cell]) in self.walls

  
class State:
    def __init__(self, cell, parent, cost):
        self.cell = cell
        self.parent = parent
        self.cost = cost
        self.ordinal = None
        self.node = None
        self.heuristic = None
        self.f = None

    def str(self):
        status = "V%d" % self.ordinal if self.ordinal else "F"
        result = "%s - (%d, %d) - g=%d" % (status, self.cell.row, self.cell.column, self.cost)
        if self.heuristic is not None:
            result += ", h=%d" % self.heuristic
        if self.f is not None:
            result += ", f=%d" % self.f
        return result

    def get_node(self):
        if not self.node:
            self.node = pydot.Node(self.str())

        return self.node

class MazeSearch:
    __metaclass__ = ABCMeta

    def __init__(self, maze):
        self.maze = maze

    def get_successor_states(self, current_state, visited):
        result = []
        cell = current_state.cell
        for (successor_cell, move_cost) in [
            (Cell(cell.row - 1, cell.column), self.costs['up']),
            (Cell(cell.row, cell.column + 1), self.costs['right']),
            (Cell(cell.row + 1, cell.column), self.costs['down']),
            (Cell(cell.row, cell.column - 1), self.costs['left'])
                ]:
            if self.maze.cell_valid(successor_cell) and not self.maze.blocked(cell, successor_cell) and self.should_successors_include(successor_cell, current_state):
                result.append(State(successor_cell, current_state, current_state.cost + move_cost))

        return result

    def should_successors_include(self, cell, state):
        return cell not in self.visited

    def should_expand(self, state):
        return state.cell not in self.visited

    @abstractmethod
    def initialize_algorithm(self):
        pass

    @abstractmethod
    def get_frontier_states(self):
        pass

    @abstractmethod
    def get_state_to_expand(self):
        pass

    @abstractmethod
    def add_successors_to_fringe(self, state, successors):
        pass

    def state_expanded(self, state):
        self.visited.append(state.cell)

    def search(self):
        self.graph = pydot.Dot()
        self.visited = []
        self.cells = defaultdict(lambda: [])
        expanded_count = 0

        self.initialize_algorithm()

        while len(self.get_frontier_states()) > 0:
            state = self.get_state_to_expand()
            if not self.should_expand(state):
                continue

            self.state_expanded(state)

            expanded_count += 1
            state.ordinal = expanded_count
          
            self.graph.add_node(state.get_node())
            
            if state.parent is not None:
                self.graph.add_edge(pydot.Edge(state.parent.get_node().get_name(), state.get_node().get_name(), 
                    label=" " + str(state.cost - state.parent.cost) + " ", labelfloat=False))


            self.cells[state.cell].append(state)

            if state.cell == self.maze.goal:
                for fstate in self.get_frontier_states():
                    self.cells[fstate.cell].append(fstate)
                    self.graph.add_node(fstate.get_node())
                    fstate.get_node().set('color', 'gray')
                    self.graph.add_edge(pydot.Edge(fstate.parent.get_node().get_name(), fstate.get_node().get_name(), 
                        label=" " + str(fstate.cost - fstate.parent.cost) + " ", labelfloat=False))

                path_state = state
                while path_state is not None:
                    path_state.node.set('color', 'red')
                    path_state = path_state.parent
                
                return state

            else:
                successors = self.get_successor_states(state, self.visited)
                self.add_successors_to_fringe(state, successors)

        return None


class DFS(MazeSearch):
    def __init__(self, maze):
        super(DFS, self).__init__(maze)
        self.costs = {'up': 1, 'right': 1, 'down': 1, 'left': 1}

    def initialize_algorithm(self):
        self.stack = []
        self.stack.append(State(self.maze.start, None, 0))

    def get_frontier_states(self):
        return self.stack

    def get_state_to_expand(self):
        return self.stack.pop(0)

    def add_successors_to_fringe(self, state, successors):
        self.stack = successors + self.stack


class Uniform(MazeSearch):
    def __init__(self, maze):
        super(Uniform, self).__init__(maze)
        self.costs = {'up': 1, 'right': 4, 'down': 10, 'left': 2}

    def initialize_algorithm(self):
        self.list = []
        self.list.append((0, State(self.maze.start, None, 0)))

    def get_frontier_states(self):
        return [tuple[1] for tuple in self.list]

    def f(self, state):
        return state.cost

    def get_state_to_expand(self):
        f_to_tuples = defaultdict(lambda: [])
        for (f, state) in self.list:
            f_to_tuples[f].append((f, state))

        smallest_f = sorted(f_to_tuples.keys())[0]
        smallest_f_states = f_to_tuples[smallest_f]

        insertion_sorted_smallest_f_tuples = sorted(smallest_f_states, key=lambda item: self.list.index(item))

        result_tuple = insertion_sorted_smallest_f_tuples[0]
        self.list.remove(result_tuple)
        return result_tuple[1]

    def add_successors_to_fringe(self, state, successors):
        self.list = self.list + [(self.f(item), item) for item in successors]


class DFSCycleChecking(DFS):
    def should_successors_include(self, cell, state):
        return True

    def should_expand(self, state):
        path_cells = []
        path_state = state.parent
        while path_state is not None:
            path_cells.append(path_state.cell)
            path_state = path_state.parent

        return state.cell not in path_cells

class Greedy(Uniform):
    def __init__(self, maze):
        super(Greedy, self).__init__(maze)
        self.costs = {'up': 1, 'right': 1, 'down': 1, 'left': 1}

    def heuristic(self, state):
        state.heuristic = abs(state.cell.row - maze.goal.row) + abs(state.cell.column - maze.goal.column)
        return state.heuristic

    def f(self, state):
        return self.heuristic(state)

class AStar(Greedy):
    def f(self, state):
        state.f = self.heuristic(state) + state.cost
        return state.f


if __name__ == '__main__':
    maze = Maze(num_rows=5, num_columns=5, start_tuple=(3, 2), goal_tuple=(1, 2), 
            wall_tuples=(((1, 0), (1, 1)), ((2 ,1), (1, 1)), ((2, 2), (1, 2)), ((3, 1), (2, 1)), ((3, 2), (2, 2)), ((3, 2), (3, 3))))

    for algorithm in [DFS, Uniform, Greedy, AStar]:
        search = algorithm(maze)
        search.search()
        name = search.__class__.__name__
        search.graph.write('%stree.dot' % name.lower())

        print name
        print

        for cell in sorted(search.cells.keys()):
            print cell 
            for state in search.cells[cell]:
                print state.str()

        print
        print

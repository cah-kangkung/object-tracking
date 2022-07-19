from kalman_filter import KalmanFilter
import numpy as np


class Object(object):

    def __init__(self, detect , ID):
        
        self.prediction = np.asarray(detect)
        self.object_id = ID 
        self.KF = KalmanFilter(1,0.1,0.1)
        self.skip_count = 0 
        self.line = [] 
        
class ObjectTracker(object):

    def __init__(self, min_dist , max_skip ,line_length , object_id):
    
        self.min_dist  = min_dist 
        
        self.max_skip = max_skip
        
        self.line_length = line_length
        
        self.objects = []
        
        self.object_id = object_id

    def Update(self, detections):

        if self.objects ==[]:
            
            for i in range(len(detections)):
                
                self.objects.append( Object(detections[i], self.object_id) )
                
                self.object_id += 1
                
        N , M = len(self.objects), len(detections)
        
        cost_matrix = np.zeros(shape=(N, M)) 
        
        for i in range(N):
            
            for j in range(M):
                
                diff = self.objects[i].prediction - detections[j]
                
                cost_matrix[i][j] = np.sqrt(diff[0][0]*diff[0][0] +diff[1][0]*diff[1][0])

        cost_matrix = (0.5) * cost_matrix 

        assign = []
        for _ in range(N):
            assign.append(-1)
            
        rows, cols = get_minimum_cost_assignment(cost_matrix)
        
        for i in range(len(rows)):
            assign[rows[i]] = cols[i]

        unassign = []
        
        for i in range(len(assign)):
            
            if (assign[i] != -1):
                
                if (cost_matrix[i][assign[i]] > self.min_dist):
                    
                    assign[i] = -1
                    unassign.append(i)
            else:
                self.objects[i].skip_count += 1

        del_objects = []
        for i in range(len(self.objects)):
            if (self.objects[i].skip_count > self.max_skip):
                del_objects.append(i)
        if len(del_objects) > 0: 
            for id in del_objects:
                if id < len(self.objects):
                    del self.objects[id]
                    del assign[id]         

        for i in range(len(detections)):
                if i not in assign:
                    self.objects.append( Object( detections[i], self.object_id )  )
                    self.object_id += 1


                
        for i in range(len(assign)):
            self.objects[i].KF.predict()

            if(assign[i] != -1):
                self.objects[i].skip_count = 0
                self.objects[i].prediction = self.objects[i].KF.update( detections[assign[i]])
            else:
                self.objects[i].prediction = self.objects[i].KF.update( np.array([
                    [0],
                    [0],
                    [0],
                    [0],
                ]))

            if(len(self.objects[i].line) > self.line_length):
                for j in range( len(self.objects[i].line) - self.line_length):
                    del self.objects[i].line[j]

            self.objects[i].line.append(self.objects[i].prediction)
            self.objects[i].KF.lastResult = self.objects[i].prediction

class Hungarian(object):

    def __init__(self, arr_costs):
        self.X = arr_costs.copy()

        n, m = self.X.shape
        self.u_row = np.ones(n, dtype=bool)
        self.u_column = np.ones(m, dtype=bool)
        self.r_0Z = 0
        self.c_0Z = 0
        self.course = np.zeros((n + m, 2), dtype=int)
        self.check = np.zeros((n, m), dtype=int)

    def clear(self):
        self.u_row[:] = True
        self.u_column[:] = True

def row_reduction(assignment):
    assignment.X -= assignment.X.min(axis=1)[:, np.newaxis]
    for i, j in zip(*np.where(assignment.X == 0)):
        if assignment.u_column[j] and assignment.u_row[i]:
            assignment.check[i, j] = 1
            assignment.u_column[j] = False
            assignment.u_row[i] = False

    assignment.clear()
    return cover_columns

def cover_columns(assignment):
    check = (assignment.check == 1)
    assignment.u_column[np.any(check, axis=0)] = False

    if check.sum() < assignment.X.shape[0]:
        return cover_zeros

def cover_zeros(assignment):
    X = (assignment.X == 0).astype(int)
    covered = X * assignment.u_row[:, np.newaxis]
    covered *= np.asarray(assignment.u_column, dtype=int)
    n = assignment.X.shape[0]
    m = assignment.X.shape[1]

    while True:
        row, col = np.unravel_index(np.argmax(covered), (n, m))   
        if covered[row, col] == 0:
            return generate_zeros
        else:
            assignment.check[row, col] = 2
            star_col = np.argmax(assignment.check[row] == 1)
            if assignment.check[row, star_col] != 1:
                assignment.r_0Z = row
                assignment.c_0Z = col
                count = 0
                course = assignment.course
                course[count, 0] = assignment.r_0Z
                course[count, 1] = assignment.c_0Z

                while True:
                    row = np.argmax(assignment.check[:, course[count, 1]] == 1)
                    if assignment.check[row, course[count, 1]] != 1:
                        break
                    else:
                        count += 1
                        course[count, 0] = row
                        course[count, 1] = course[count - 1, 1]

                    col = np.argmax(assignment.check[course[count, 0]] == 2)
                    if assignment.check[row, col] != 2:
                        col = -1
                    count += 1
                    course[count, 0] = course[count - 1, 0]
                    course[count, 1] = col

                for i in range(count + 1):
                    if assignment.check[course[i, 0], course[i, 1]] == 1:
                        assignment.check[course[i, 0], course[i, 1]] = 0
                    else:
                        assignment.check[course[i, 0], course[i, 1]] = 1

                assignment.clear()
                assignment.check[assignment.check == 2] = 0
                return cover_columns
            else:
                col = star_col
                assignment.u_row[row] = False
                assignment.u_column[col] = True
                covered[:, col] = X[:, col] * (
                    np.asarray(assignment.u_row, dtype=int))
                covered[row] = 0

def generate_zeros(assignment):
    if np.any(assignment.u_row) and np.any(assignment.u_column):
        minimum_value = np.min(assignment.X[assignment.u_row], axis=0)
        minimum_value = np.min(minimum_value[assignment.u_column])
        assignment.X[~assignment.u_row] += minimum_value
        assignment.X[:, assignment.u_column] -= minimum_value
    return cover_zeros

def get_minimum_cost_assignment(arr_costs):
    arr_costs = np.asarray(arr_costs)

    
    if arr_costs.shape[1] < arr_costs.shape[0]:
        arr_costs = arr_costs.T
        is_T = True
    else:
        is_T = False

    assignment = Hungarian(arr_costs)


    run = None if 0 in arr_costs.shape else row_reduction

    while run is not None:
        run = run(assignment)

    if is_T:
        check = assignment.check.T
    else:
        check = assignment.check
    return np.where(check == 1)
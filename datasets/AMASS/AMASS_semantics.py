import os, sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())
    
from datasets.dataset import SkeletonSemantics


class AmassSemantics(SkeletonSemantics):
    skeleton_semantics = {
        '0': {
            'upper body': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
            'lower body': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        },
        
        '1': {
            'spine': [0, 3, 6, 9],
            'head' : [12, 15, 22, 23],
            'left arm' : [14, 17, 19, 21],
            'right arm': [13, 16, 18, 20],
            'left leg' : [1, 4, 7, 10],
            'right leg': [2, 5, 8, 11]
        },
        
        '2': {            
            'neck': [12],  'head': [15], 'hair': [22], 'hat': [23],
            'hip': [0],  'spine': [3],  'spine+': [6],  'throax': [9],
            
            'lcollar': [14],  'lshoudler': [17],  'lelow': [19], 'larm': [21],
            'rcollar': [13],  'rshoudler': [16],  'relow': [18], 'rarm': [20],
            
            'lhip': [1],  'lknee': [4],  'lankle': [7], 'lfoot': [10],
            'rhip': [2],  'rknee': [5],  'rankle': [8], 'rfoot': [11],
        },
    }
    def __init__(self):
        super().__init__(self.skeleton_semantics)
    
    def check_level_dims(self, dims):
        assert dims <= 1
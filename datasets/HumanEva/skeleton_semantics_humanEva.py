import os, sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())

from datasets.dataset import SkeletonSemantics


class HumanEva18nodeSkeletonSemantics(SkeletonSemantics):
    skeleton_semantics = {
        '0': {
            'upper body': [1, 14, 15, 2, 3, 4, 5, 6, 7],
            'lower body': [0, 16, 17, 8, 9, 10, 11, 12, 13]
        },
        
        '1': {
            'tail': [0, 16, 17],
            'head' : [1, 14, 15],
            'left arm' : [2, 3, 4],
            'right arm': [5, 6, 7],
            'left leg' : [8, 9, 10],
            'right leg': [11, 12, 13]
        },
        
        '2': {
            'hip': [0],  'tail': [16],  'tail+': [17],
            
            'rhip': [11],  'rknee': [12],  'rfoot': [13],
            'lhip': [8],   'lknee': [8],   'lfoot': [10],

            'rshoulder': [5],  'relbow': [6],  'rwrist': [7],
            'lshoulder': [2],  'lelbow': [3],  'lwrist': [4],

            'thorax': [1],  'neck': [14],  'head': [15]
        }
    }
    def __init__(self):
        super().__init__(self.skeleton_semantics)
    
    def check_level_dims(self, dims):
        assert dims <= 1
        

class HumanEva15nodeSkeletonSemantics(SkeletonSemantics):
    skeleton_semantics = {
        '0': {
            'upper body': [14, 15, 2, 3, 4,  5,  6,  7],
            'lower body': [0,  1,  8, 9, 10, 11, 12, 13]
        },
        
        '1': {
            'tail' : [0, 1, 14],    # overlapping with 'head'
            'head' : [1, 14, 15],   # overlapping with 'tail'
            'left arm' : [2, 3, 4],
            'right arm': [5, 6, 7],
            'left leg' : [8, 9, 10],
            'right leg': [11, 12, 13]
        },
        
        '2': {
            'hip': [0],
            
            'rhip': [11],  'rknee': [12],  'rfoot': [13],
            'lhip': [8],   'lknee': [8],   'lfoot': [10],

            'rshoulder': [5],  'relbow': [6],  'rwrist': [7],
            'lshoulder': [2],  'lelbow': [3],  'lwrist': [4],

            'thorax': [1],  'neck': [14],  'head': [15]
        }
    }
    def __init__(self):
        super().__init__(self.skeleton_semantics)
    
    def check_level_dims(self, dims):
        assert dims <= 1



HumanEva18expmapSkeletonSemantics = HumanEva18nodeSkeletonSemantics
HumanEva18posSkeletonSemantics = HumanEva18nodeSkeletonSemantics

HumanEva15expmapSkeletonSemantics = HumanEva15nodeSkeletonSemantics
HumanEva15posSkeletonSemantics = HumanEva15nodeSkeletonSemantics
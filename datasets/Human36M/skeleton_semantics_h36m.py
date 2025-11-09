import os, sys
if __name__ == "__main__":
    sys.path.append(os.getcwd())

from datasets.dataset import SkeletonSemantics



class Human36m18nodeSkeletonSemantics(SkeletonSemantics):
    skeleton_semantics = {
        '0': {
            'upper body': [9, 10, 11, 12, 13, 14, 15, 16, 17],
            'lower body': [0, 7, 8, 1, 2, 3, 4, 5, 6]
        },
        
        '1': {
            'spine': [0, 7, 8],
            'head' : [9, 10, 11],
            'left arm' : [12, 13, 14],
            'right arm': [15, 16, 17],
            'left leg' : [4, 5, 6],
            'right leg': [1, 2, 3]
        },
        
        '2': {
            'hip': [0],  'spine': [7],  'spine+': [8],
            
            'rhip': [1],  'rknee': [2],  'rfoot': [3],
            'lhip': [4],  'lknee': [5],  'lfoot': [6],

            'rshoulder': [15],  'relbow': [16],  'rwrist': [17],
            'lshoulder': [12],  'lelbow': [13],  'lwrist': [14],

            'thorax': [9],  'neck': [10],  'head': [11]
        },

        '3': {
            'hip0': [0],  'hip1': [1],  'hip2': [2],   
            'spine0': [21],  'spine1': [22],  'spine2': [23],
            'spine+0': [24],  'spine+1': [25],  'spine+2': [26], 
            
            'rhip0': [3],  'rhip1': [4],  'rhip2': [5],  
            'rknee0': [6],  'rknee1': [7],  'rknee2': [8],  
            'rfoot0': [9],  'rfoot1': [10],  'rfoot2': [11],  
            
            'lhip0': [12],  'lhip1': [13],  'lhip2': [14],  
            'lknee0': [15],  'lknee1': [16],  'lknee2': [17],  
            'lfoot0': [18],  'lfoot1': [19],  'lfoot2': [20],  

            'rshoulder0': [45], 'rshoulder1': [46], 'rshoulder2': [47],
            'relbow0': [48],  'relbow1': [49],  'relbow2': [50],  
            'rwrist0': [51],  'rwrist1': [52],  'rwrist2': [53],  
             
            'lshoulder0': [36], 'lshoulder1': [37], 'lshoulder2': [38],
            'lelbow0': [39],  'lelbow1': [40],  'lelbow2': [41],  
            'lwrist0': [42],  'lwrist1': [43],  'lwrist2': [44], 
            
            'thorax0': [27],  'thorax1': [28],  'thorax2': [29],
            'neck0': [30],  'neck1': [31],  'neck2': [32],  
            'head0': [33],  'head1': [34],  'head2': [35],  
        }
    }
    def __init__(self):
        super().__init__(self.skeleton_semantics)
    
    def check_level_dims(self, dims):
        assert dims <= 1


Human36m18posSkeletonSemantics = Human36m18nodeSkeletonSemantics
Human36m18expmapSkeletonSemantics = Human36m18nodeSkeletonSemantics



class Human36m24posSkeletonSemantics(SkeletonSemantics):
    skeleton_semantics = {
        '0': {
            'upper body': [10, 11, 12, 23, 14, 15, 16, 17, 19, 20, 21, 22],
            'lower body': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 18]
        },
        
        '1': {
            'spine': [0, 9, 13, 18],
            'head' : [10, 11, 12, 23],
            'left arm' : [14, 15, 16, 17],
            'right arm': [19, 20, 21, 22],
            'left leg' : [5, 6, 7, 8],
            'right leg': [1, 2, 3, 4]
        },
        
        '2': {
            'hip': [0],  'spine': [9],  'lshouder': [13],  'rshouder': [18],
            
            'rhip': [1],  'rknee': [2],  'rfoot': [3],  'rfoot+': [4],
            'lhip': [5],  'lknee': [6],  'lfoot': [7],  'lfoot+': [8],

            'relbow': [19],  'rwrist': [20],  'rhand': [21], 'rhand+': [22],
            'lelbow': [14],  'lwrist': [15],  'lhand': [16], 'lhand+': [17],

            'thorax': [10],  'neck': [11],  'head': [12], 'hair': [23]
        },
        
        '3': {
            'hip0': [0],  'hip1': [1],  'hip2': [2],   
            'spine0': [27],  'spine1': [28],  'spine2': [29],
            'lshoulder0': [39],  'lshoulder1': [40],  'lshoulder2': [41], 
            'rshoulder0': [54],  'rshoulder1': [55],  'rshoulder2': [56],  
            
            'rhip0': [3],  'rhip1': [4],  'rhip2': [5],  
            'rknee0': [6],  'rknee1': [7],  'rknee2': [8],  
            'rfoot0': [9],  'rfoot1': [10],  'rfoot2': [11],  
            'rfoot+0': [12],  'rfoot+1': [13],  'rfoot+2': [14],
            
            'lhip0': [15],  'lhip1': [16],  'lhip2': [17],  
            'lknee0': [18],  'lknee1': [19],  'lknee2': [20],  
            'lfoot0': [21],  'lfoot1': [22],  'lfoot2': [23],  
            'lfoot+0': [24],  'lfoot+1': [25],  'lfoot+2': [26],

            'relbow0': [57],  'relbow1': [58],  'relbow2': [59],  
            'rwrist0': [60],  'rwrist1': [61],  'rwrist2': [62],  
            'rhand0': [63],  'rhand1': [64],  'rhand2': [65],
            'rhand+0': [66],  'rhand+1': [67],  'rhand+2': [68],
             
            'lelbow0': [42],  'lelbow1': [43],  'lelbow2': [44],  
            'lwrist0': [45],  'lwrist1': [46],  'lwrist2': [47],  
            'lhand0': [48],  'lhand1': [49],  'lhand2': [50],
            'lhand+0': [51],  'lhand+1': [52],  'lhand+2': [53],

            'thorax0': [30],  'thorax1': [31],  'thorax2': [32],
            'neck0': [33],  'neck1': [34],  'neck2': [35],  
            'head0': [36],  'head1': [37],  'head2': [38],  
            'hair0': [69],  'hair1': [70],  'hair2': [71], 
        }
    }
    def __init__(self):
        super().__init__(self.skeleton_semantics)
    
    def check_level_dims(self, dims):
        assert dims <= 1

  
class Human36m24expmapSkeletonSemantics(SkeletonSemantics):
    skeleton_semantics = {
        '0': {
            'upper body': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
            'lower body': [0, 9, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8]
        },
        
        '1': {
            'spine': [0, 9, 10, 11],
            'head' : [12, 13, 22, 23],
            'left arm' : [14, 15, 16, 17],
            'right arm': [18, 19, 20, 21],
            'left leg' : [5, 6, 7, 8],
            'right leg': [1, 2, 3, 4]
        },
        
        '2': {
            'hip': [0],  'spine': [9],  'spine+': [10],  'thorax': [11],
            
            'rhip': [1],  'rknee': [2],  'rfoot': [3],  'rfoot+': [4],
            'lhip': [5],  'lknee': [6],  'lfoot': [7],  'lfoot+': [8],

            'rshoulder': [18],  'relbow': [19],  'rwrist': [20],  'rhand': [21],
            'lshoulder': [14],  'lelbow': [15],  'lwrist': [16],  'lhand': [17],

            'neck': [12],  'head': [13],  'hair': [22], 'hat': [23]
        },
        
        '3': {
            'hip0': [0],  'hip1': [1],  'hip2': [2],   
            'spine0': [27],  'spine1': [28],  'spine2': [29],  
            'spine+0': [30],  'spine+1': [31],  'spine+2': [32],  
            'thorax0': [33],  'thorax1': [34],  'thorax2': [35],
            
            'rhip0': [3],  'rhip1': [4],  'rhip2': [5],  
            'rknee0': [6],  'rknee1': [7],  'rknee2': [8],  
            'rfoot0': [9],  'rfoot1': [10],  'rfoot2': [11],  
            'rfoot+0': [12],  'rfoot+1': [13],  'rfoot+2': [14],
            
            'lhip0': [15],  'lhip1': [16],  'lhip2': [17],  
            'lknee0': [18],  'lknee1': [19],  'lknee2': [20],  
            'lfoot0': [21],  'lfoot1': [22],  'lfoot2': [23],  
            'lfoot+0': [24],  'lfoot+1': [25],  'lfoot+2': [26],

            'rshoulder0': [54],  'rshoulder1': [55],  'rshoulder2': [56],  
            'relbow0': [57],  'relbow1': [58],  'relbow2': [59],  
            'rwrist0': [60],  'rwrist1': [61],  'rwrist2': [62],  
            'rhand0': [63],  'rhand1': [64],  'rhand2': [65],
            
            'lshoulder0': [42],  'lshoulder1': [43],  'lshoulder2': [44],  
            'lelbow0': [45],  'lelbow1': [46],  'lelbow2': [47],  
            'lwrist0': [48],  'lwrist1': [49],  'lwrist2': [50],  
            'lhand0': [51],  'lhand1': [52],  'lhand2': [53],

            'neck0': [36],  'neck1': [37],  'neck2': [38],  
            'head0': [39],  'head1': [40],  'head2': [41],  
            'hair0': [66],  'hair1': [67],  'hair2': [68], 
            'hat0': [69],  'hat1': [70],  'hat2': [71]
        }
    }
    def __init__(self):
        super().__init__(self.skeleton_semantics)
    
    def check_level_dims(self, dims):
        assert dims <= 1



class Human36m22posSkeletonSemantics(SkeletonSemantics):
    skeleton_semantics = {        
        '2': {
            'spine': [8],  'lshouder': [12],  'rshouder': [17],
            
            'rhip': [0],  'rknee': [1],  'rfoot': [2],  'rfoot+': [3],
            'lhip': [4],  'lknee': [5],  'lfoot': [6],  'lfoot+': [7],

            'relbow': [18],  'rwrist': [19],  'rhand': [20], 'rhand+': [21],
            'lelbow': [13],  'lwrist': [14],  'lhand': [15], 'lhand+': [16],

            'thorax': [9],  'neck': [10],  'head': [11]
        },
        
        '3': {
            'spine0': [24],  'spine1': [25],  'spine2': [26],
            'lshoulder0': [36],  'lshoulder1': [37],  'lshoulder2': [38],
            'rshoulder0': [51],  'rshoulder1': [52],  'rshoulder2': [53],
            
            'rhip0': [0],  'rhip1': [1],  'rhip2': [2],
            'rknee0': [3],  'rknee1': [4],  'rknee2': [5],  
            'rfoot0': [6],  'rfoot1': [7],  'rfoot2': [8],  
            'rfoot+0': [9],  'rfoot+1': [10],  'rfoot+2': [11], 
             
            'lhip0': [12],  'lhip1': [13],  'lhip2': [14],
            'lknee0': [15],  'lknee1': [16],  'lknee2': [17],  
            'lfoot0': [18],  'lfoot1': [19],  'lfoot2': [20],  
            'lfoot+0': [21],  'lfoot+1': [22],  'lfoot+2': [23], 
        
            'relbow0': [54],  'relbow1': [55],  'relbow2': [56],  
            'rwrist0': [57],  'rwrist1': [58],  'rwrist2': [59],  
            'rhand0': [60],  'rhand1': [61],  'rhand2': [62],  
            'rhand+0': [63],  'rhand+1': [64],  'rhand+2': [65],
             
            'lelbow0': [39],  'lelbow1': [40],  'lelbow2': [41], 
            'lwrist0': [42],  'lwrist1': [43],  'lwrist2': [44],  
            'lhand0': [45],  'lhand1': [46],  'lhand2': [47],  
            'lhand+0': [48],  'lhand+1': [49],  'lhand+2': [50],

            'thorax0': [27],  'thorax1': [28],  'thorax2': [29],
            'neck0': [30],  'neck1': [31],  'neck2': [32],
            'head0': [33],  'head1': [34],  'head2': [35],  
        }
    }
    def __init__(self):
        super().__init__(self.skeleton_semantics)
    
    def check_level_dims(self, dims):
        assert dims <= 1
        


class Human36m23posSkeletonSemantics(SkeletonSemantics):
    skeleton_semantics = {        
        '2': {
            'hip': [0],  'spine': [9],  'lshouder': [13],  'rshouder': [18],
            
            'rhip': [1],  'rknee': [2],  'rfoot': [3],  'rfoot+': [4],
            'lhip': [5],  'lknee': [6],  'lfoot': [7],  'lfoot+': [8],

            'relbow': [19],  'rwrist': [20],  'rhand': [21], 'rhand+': [22],
            'lelbow': [14],  'lwrist': [15],  'lhand': [16], 'lhand+': [17],

            'thorax': [10],  'neck': [11],  'head': [12]
        },
        
        '3': {
            'hip0': [0],  'hip1': [1],  'hip2': [2],   
            'spine0': [27],  'spine1': [28],  'spine2': [29],
            'lshoulder0': [39],  'lshoulder1': [40],  'lshoulder2': [41], 
            'rshoulder0': [54],  'rshoulder1': [55],  'rshoulder2': [56],  
            
            'rhip0': [3],  'rhip1': [4],  'rhip2': [5],  
            'rknee0': [6],  'rknee1': [7],  'rknee2': [8],  
            'rfoot0': [9],  'rfoot1': [10],  'rfoot2': [11],  
            'rfoot+0': [12],  'rfoot+1': [13],  'rfoot+2': [14],
            
            'lhip0': [15],  'lhip1': [16],  'lhip2': [17],  
            'lknee0': [18],  'lknee1': [19],  'lknee2': [20],  
            'lfoot0': [21],  'lfoot1': [22],  'lfoot2': [23],  
            'lfoot+0': [24],  'lfoot+1': [25],  'lfoot+2': [26],

            'relbow0': [57],  'relbow1': [58],  'relbow2': [59],  
            'rwrist0': [60],  'rwrist1': [61],  'rwrist2': [62],  
            'rhand0': [63],  'rhand1': [64],  'rhand2': [65],
            'rhand+0': [66],  'rhand+1': [67],  'rhand+2': [68],
             
            'lelbow0': [42],  'lelbow1': [43],  'lelbow2': [44],  
            'lwrist0': [45],  'lwrist1': [46],  'lwrist2': [47],  
            'lhand0': [48],  'lhand1': [49],  'lhand2': [50],
            'lhand+0': [51],  'lhand+1': [52],  'lhand+2': [53],

            'thorax0': [30],  'thorax1': [31],  'thorax2': [32],
            'neck0': [33],  'neck1': [34],  'neck2': [35],  
            'head0': [36],  'head1': [37],  'head2': [38]
        }
    }
    def __init__(self):
        super().__init__(self.skeleton_semantics)
    
    def check_level_dims(self, dims):
        assert dims <= 1
    
   
    
class Human36m16nodeSkeletonSemantics(SkeletonSemantics):
    skeleton_semantics = {
        '0': {
            'upper body': [8, 9, 10, 11, 12, 13, 14, 15],
            'lower body': [0, 7, 1, 2, 3, 4, 5, 6]
        },
                
        '2': {
            'hip': [0],  'spine': [7],
            
            'rhip': [1],  'rknee': [2],  'rfoot': [3],
            'lhip': [4],  'lknee': [5],  'lfoot': [6],

            'relbow': [14],  'rwrist': [15],
            'lshoulder': [11],  'lelbow': [12],  'lwrist': [13],

            'thorax': [8],  'neck': [9],  'head': [10]
        },
    }
    def __init__(self):
        super().__init__(self.skeleton_semantics)
    
    def check_level_dims(self, dims):
        assert dims <= 1
class PriorityQueue(object):
    def __init__(self, n):
        self.n = n
        self.list = []
    def getList(self):
        return self.list
    def AddElement(self,x):
        self.list.append(x)
        self.list.sort(reverse=True) 
        if len(self.list)>3:
            self.list.pop(len(self.list)-1)
        

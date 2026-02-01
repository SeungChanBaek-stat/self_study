class Node():
    def __init__(self):
        self.data = None
        self.link = None
        
def printNodes(start):
    current = start
    if current.link == None:
        return
    print(current.data, end = ' ')
    while current.link != start:
        current = current.link
        print(current.data, end = ' ')    
    print()
    
def insertNodes(findData, insertData):
    global memory, head, current, pre
    
    if head.data == findData:
        node = Node()
        node.data = insertData
        node.link = head
        last = head
        while last.link != head:
            last = last.link
        last.link = node
        head = node
        return
    
    current = head
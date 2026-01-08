class Node:
    def __init__(self):
        self.data = None
        self.link = None

class SimpleLinkedListTest:
    def __init__(self, data_array : list,):
        self.data_array = data_array
        self.memory = list()
        self.head = None
        
        
        pass
    
    def print_linked_list(self,):
        print("\n")
        curr_node = self.head
        print(curr_node.data, end = " ")
        while curr_node.link != None:
            curr_node = curr_node.link
            print(curr_node.data, end = " ")
            # if curr_node.link == None:
            #     break
        print("\n")
        return None
            
    
    def link_standard(self, ):
        for i in range(len(self.data_array)):
            if i == 0:
                new_node = Node()
                new_data = self.data_array[i]
                new_node.data = new_data
                self.memory.append(new_node)
                self.head = new_node
            else:
                old_node = new_node
                new_node = Node()
                new_data = self.data_array[i]
                new_node.data = new_data
                old_node.link = new_node
                self.memory.append(new_node)
                
        return None
        
        
    # 삽입하려는 데이터는 삽입지점데이터보다 1칸앞에 삽입됨
    def add_data(self, new_data, old_data):
        curr_node = self.head
        
        # 삽입지점이 맨 처음인 경우
        if curr_node.data == old_data:
            new_node = Node()
            new_node.data = new_data
            new_node.link = curr_node
            self.head = new_node
            self.data_array.append(new_data)
            # print(f"You are here : {self.data_array}")
            
            return None
        
        # 삽입지점이 연결리스트 중간인 경우
        elif old_data in self.data_array:
            while curr_node.data != old_data:
                pre_node = curr_node
                curr_node = curr_node.link
            new_node = Node()
            new_node.data = new_data
            new_node.link = curr_node
            pre_node.link = new_node
            self.data_array.append(new_data)
            
            return None
        
        # 삽입지점이 연결리스트 마지막인 경우 (old_data가 기존 연결리스트에 없는 경우)
        else:
            # print("You are here")
            while curr_node.link != None:
                pre_node = curr_node
                curr_node = curr_node.link
            # print(curr_node.data)
            new_node = Node()
            new_node.data = new_data
            curr_node.link = new_node
            self.data_array.append(new_data)
            
            return None
        
    def del_data(self, del_data):
        curr_node = self.head
        if curr_node.data == del_data:
            next_node = curr_node.link
            self.head = next_node
            del curr_node
            
            return None
        else:
            while curr_node.data != del_data:
                pre_node = curr_node
                curr_node = curr_node.link
            next_node = curr_node.link
            pre_node.link = next_node
            del curr_node
            
            return None
        
    def search_data(self, search_data):
        curr_node = self.head
        while curr_node.link != None:
            curr_node = curr_node.link
            if curr_node.data == search_data:
                return curr_node
        
        return None
        
        
        
        
        
        



if __name__ == "__main__":
    data_test = ["Musk", "Tibshirani", "Bayes", "LeCam", "Kartmen"]

    simplelinkedlist_test = SimpleLinkedListTest(data_test)

    simplelinkedlist_test.link_standard()
    simplelinkedlist_test.print_linked_list()
    # simplelinkedlist_test.add_new("Sutskever", "Musk")
    # simplelinkedlist_test.add_new("Sutskever", "Bayes")
    # simplelinkedlist_test.add_new("Sutskever", "Kartmen")
    simplelinkedlist_test.add_data("Sutskever", "new")
    simplelinkedlist_test.print_linked_list()
    
    # simplelinkedlist_test.del_node("Musk")
    # simplelinkedlist_test.del_node("LeCam")
    simplelinkedlist_test.del_data("Sutskever")
    simplelinkedlist_test.print_linked_list()
    
    searched_node = simplelinkedlist_test.search_data("Tibshirani")
    if searched_node != None:
        print(searched_node.data)
    else:
        print("None")
    
    searched_node = simplelinkedlist_test.search_data("Sutskever")
    if searched_node != None:
        print(searched_node.data)
    else:
        print("None")
    
    # for i in range(len(simplelinkedlist_test.memory)):
    #     curr_node = simplelinkedlist_test.memory[i]
    #     print(curr_node.data)
    #     curr_node = curr_node.link
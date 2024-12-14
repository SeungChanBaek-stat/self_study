class LinearList:
    def __init__(self, list:list, select:int):
        self.list = list
        self.select = select # 1:추가, 2:삽입, 3:삭제, 4:종료
        pass

    def add_data(self, item:str):
        self.list.append(None)
        N = len(self.list)
        self.list[N-1] = item

        return self.list
    
    def insert_data(self, position:int, item:str):
        if position < 0 or position > len(self.list):
            print("데이터를 삽입할 범위를 벗어났습니다.")
            return
        
        self.list.append(None)
        kLen = len(self.list)

        for i in range(kLen-1, position, -1):
            self.list[i] = self.list[i-1]
            self.list[i-1] = None

        self.list[position] = item

        return self.list
    
    def delete_data(self, position:int):
        if position < 0 or position >= len(self.list):
            print("데이터를 삭제할 범위를 벗어났습니다.")
            return

        kLen = len(self.list)
        self.list[position] = None

        for i in range(position+1, kLen):
            self.list[i-1] = self.list[i]
            self.list[i] = None

        del(self.list[kLen-1])

        return self.list
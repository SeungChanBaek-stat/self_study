def add_data(katok : list, friend : str):
    katok.append(None)
    N = len(katok)
    katok[N-1] = friend
    
    return katok
import sys
import os
import json

sys.path.insert(0, 'src')

def main(targets):
    
    a = 1
    
    if 'test' in targets:
        print(a)
        
        if 'visualize' in targets:
            print(a + 3)
    
    if 'reset' in targets:
        a = None

if __name__ == '__main__':
    # run via:
    # python main.py data features model
    targets = sys.argv[1:]
    main(targets)
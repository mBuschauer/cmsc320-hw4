import libs.PlanetTerp as planetterp
import pandas as pd       
from tqdm import tqdm   
import time  

if __name__ == "__main__":
    all_profs = []
    limit = 100
    offset = 0


    pbar = tqdm(desc="Downloading professors", unit="rows")
    while True:
        profs = planetterp.professors(
            type_="professor",
            limit=limit,
            reviews=True,
            offset=offset
        )

        if not profs:
            break
        

        all_profs.extend(profs)

        pbar.update(len(profs))

        offset += limit
        time.sleep(0.1)

    df = pd.DataFrame(all_profs)

    df.to_parquet("data/professors.parquet", index=False)


    print(f"DataFrame shape: {df.shape}")
    print("Saved to professors.parquet")
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append(".")
from doris_vtec.stage1_obs2stec.pipeline import iter_blocks, process_block
from doris_vtec.stage2_stec2vtec.pipeline import process_day
from doris_vtec.config import DEFAULT_STAGE2_CONFIG

satellite_list = ["ja3", "CRYOSAT-2", "SARAL"]


def main():
    t0 = time.time()

    # -------------------------------------------------
    # Global parameters
    # -------------------------------------------------
    start_date = pd.Timestamp(2018, 1, 1)
    proc_days = 11
    block_days = 11

    satellite = satellite_list[0]
    end_date = start_date + pd.Timedelta(days=proc_days - 1)

    sinex_path = Path("/Volumes/T7Shield/DORISData/DORISInput/sinex/dpod2020_current.snx")
    sp3_root = Path(f"/Volumes/T7Shield/DORISData/DORISInput/sp3/{satellite}")
    rinex_root = Path(f"/Volumes/T7Shield/DORISData/DORISInput/rinexobs/{satellite}")
    obs_root = Path(f"/Volumes/T7Shield/DORISData/DORISObsStorage/{satellite}")

    gim_root = Path("/Volumes/T7Shield/DORISData/DORISInput/IGSGIM")
    vtec_root = Path(f"/Volumes/T7Shield/DORISData/DORISVTECStorage/{satellite}")

    # =================================================
    # =============== STAGE 1 =========================
    # =================================================

    # pbar1 = tqdm(desc="Stage1 OBS", unit="day")

    # for block_start, block_end in iter_blocks(
    #     start_date, end_date, block_days
    # ):
    #     process_block(
    #         block_start,
    #         block_end,
    #         sinex_file=sinex_path,
    #         sp3_root=sp3_root,
    #         rinex_root=rinex_root,
    #         output_root=obs_root,
    #         satellite=satellite,
    #     )

    #     n_days = (block_end - block_start).days + 1
    #     pbar1.update(n_days)

    # pbar1.close()

    # =================================================
    # =============== STAGE 2 =========================
    # =================================================

    pbar2 = tqdm(
        pd.date_range(start_date, end_date, freq="D"),
        desc="Stage2 VTEC",
        unit="day",
    )

    for date in pbar2:
        process_day(
            date=date,
            obs_root=obs_root,
            gim_root=gim_root,
            out_root=vtec_root,
            satellite=satellite,
            config=DEFAULT_STAGE2_CONFIG,
        )

    pbar2.close()

    print(f"\nFinished in {(time.time() - t0) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
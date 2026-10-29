import h5py
import os
import time
from dataclasses import dataclass, field
from natsort import natsorted
import numpy as np
from pathlib import Path
from src.utilities.helper import savelist

MAXIMUM_SLICES_MOTION = {
    "ExamXXXX": 60,
}

@dataclass
class Splitter:
    data_dir: Path
    data_dir_processed: Path
    reg_lambda: str
    reco_name: str
    ignored_files: set[str] = field(default_factory=set)
    test_files: set[str] = field(default_factory=set)
    validation_files: set[str] = field(default_factory=set)
    crop: int = 0  # pixels cropped on each side

    def __post_init__(self):
        self.studies = sorted([x for x in self.data_dir.iterdir() if x.is_dir()])

    def _should_keep_slice(self, study_name: str, slice_idx: int, n_slices: int, crop_lower_end: bool) -> bool:
        if not crop_lower_end:
            return True
        max_slice = MAXIMUM_SLICES_MOTION.get(study_name, n_slices)  # fallback if not in dict
        lower = int(n_slices * 0.2)
        return (slice_idx > lower) and (slice_idx <= max_slice)

    def extract_data(self, study: Path, mode: str, split_list: list[str], saving: bool, crop_lower_end: bool) -> list[str]:
        out_dir = self.data_dir_processed / mode
        out_dir.mkdir(parents=True, exist_ok=True)

        file_names = natsorted(study.glob(f"**/{self.reco_name}/*{self.reg_lambda}*.mat"))
        if not file_names:
            print(f"[WARN] No files found for {study.name} in mode={mode}")
            return split_list

        print("#####################")
        print(study)

        n_files = len(file_names)
        c = self.crop

        for file_name in file_names:
            # parse slice index safely
            slice_str = file_name.stem.rsplit("_", 1)[-1]
            try:
                slice_idx = int(slice_str)
            except ValueError:
                print(f"[WARN] Could not parse slice index from {file_name.name}; skipping.")
                continue

            if not self._should_keep_slice(study.name, slice_idx, n_files, crop_lower_end):
                print(f"Slice skipped: {study.name} slice {slice_idx}")
                continue

            out_name = f"{study.name}_slice_{slice_idx}"
            split_list.append(out_name)

            if not saving:
                continue

            with h5py.File(file_name, "r") as hf:
                
                t0 = time.time()
                if c == 0:
                    rnufft = np.asarray(hf["rnufft"])
                    coilwise_rnufft = np.asarray(hf["coilwise_rnufft"])
                    grasp = np.asarray(hf["x"])
                else:
                    rnufft = np.asarray(hf["rnufft"][:, c:-c, c:-c])
                    grasp = np.asarray(hf["x"][:, c:-c, c:-c])
                    coilwise_rnufft = np.asarray(hf["coilwise_rnufft"][:, :, c:-c, c:-c])

                print(f"Loading took {time.time()-t0:.3f}s, shape={rnufft.shape}")

            np.savez(out_dir / out_name, rnufft=rnufft, coilwise_rnufft=coilwise_rnufft, grasp=grasp)

        print("#####################")
        return split_list

    def create_splits(self, saving: bool, crop_lower_end: bool):
        train, val, test = [], [], []

        for study in self.studies:
            name = study.name
            if name in self.ignored_files:
                print(f"Ignore study {name}.")
                continue

            if name in self.test_files:
                    self.extract_data(study, "test", test, saving=saving, crop_lower_end=crop_lower_end)
            elif name in self.validation_files:
                self.extract_data(study, "validation", val, saving=saving, crop_lower_end=crop_lower_end)
            else:
                self.extract_data(study, "training", train, saving=saving, crop_lower_end=crop_lower_end)

        self.data_dir_processed.mkdir(parents=True, exist_ok=True)
        savelist(self.data_dir_processed / "training.pkl", train)
        savelist(self.data_dir_processed / "validation.pkl", val)
        savelist(self.data_dir_processed / "test.pkl", test)

        print("training list len", len(train))
        print("validation list len", len(val))
        print("testing list len", len(test))


if __name__ == "__main__":
    path = Path(os.environ["DATA_DIR_RECOS"])
    path_processed = Path(os.environ["DATA_DIR_EXP"])
    DATASET = "gyn"   # "motion" or "gyn"  or "liver"
    saving = True

    if DATASET == "motion":
        # MOTION DATA
        ignored = {
            "ExamXXXX"
        }
        test = {"ExamXXXX"}
        val = {"ExamXXXX3"}
        reg_lambda = "20"
        reco_name = "FISTA_results_complete_julia"
        crop_lower_end = True
    elif DATASET == "gyn":
        # GYN DATA
        ignored = {"StudyXXXX"}
        test = {"Study1234"}
        val = {"Study2345"}
        reg_lambda = "02"
        reco_name = "FISTA_results_complete_julia"
        crop_lower_end = False
        crop = 72
    else:
        raise ValueError(f"Unknown DATASET={DATASET!r}")

    data = Splitter(
        data_dir=path,
        data_dir_processed=path_processed,
        reg_lambda=reg_lambda,
        reco_name=reco_name,
        ignored_files=ignored,
        test_files=test,
        validation_files=val,
        crop=crop
    )
    data.create_splits(saving=saving, crop_lower_end=crop_lower_end)
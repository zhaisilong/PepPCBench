from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from dataclasses import asdict

@dataclass
class ProteinConfig:
    id: str
    sequence: str

    # MSA part
    unpaired_msa: Optional[str] = None
    paired_msa: Optional[str] = None
    templates: Optional[List] = None

    # 非标准氨基酸
    modifications: Optional[List] = None


@dataclass
class AF3Config:
    name: str
    modelSeeds: List[int]
    sequences: List[ProteinConfig]
    dialect: str
    version: int

    def to_dict(self):
        # Convert the dataclass to a dictionary, handling nested Proteins
        return {
            "name": self.name,
            "modelSeeds": self.modelSeeds,
            "sequences": [
                {
                    "protein": {
                        "id": p.id,
                        "sequence": p.sequence,
                        **(
                            {"unpairedMsa": p.unpaired_msa}
                            if p.unpaired_msa is not None
                            else {}
                        ),
                        **(
                            {"pairedMsa": p.paired_msa}
                            if p.paired_msa is not None
                            else {}
                        ),
                        **(
                            {"templates": p.templates}
                            if p.templates is not None
                            else {}
                        ),
                        **(
                            {"modifications": p.modifications}
                            if p.modifications is not None
                            else {}
                        ),
                    }
                }
                for p in self.sequences
            ],
            "dialect": self.dialect,
            "version": self.version,
        }

    @staticmethod
    def from_dict(config_dict):
        return AF3Config(
            name=config_dict["name"],
            modelSeeds=config_dict["modelSeeds"],
            sequences=config_dict["sequences"],
            dialect=config_dict["dialect"],
            version=config_dict["version"],
        )


################################## Helix #####################


@dataclass
class HelixProteinConfig:
    sequence: str
    type: str = "protein"
    count: int = 1
    modification: Optional[List[Dict]] = None


@dataclass
class HelixConfig:
    job_name: str
    recycle: int = 10
    ensemble: int = 1
    entities: List[HelixProteinConfig] = field(default_factory=list)

    def to_dict(self):  # 修复后的方法
        return {
            "job_name": self.job_name,
            "recycle": self.recycle,
            "ensemble": self.ensemble,
            "entities": [
                {
                    "type": entity.type,
                    "sequence": entity.sequence,
                    "count": entity.count,
                    **(
                        {"modification": entity.modification}
                        if entity.modification is not None
                        else {}
                    ),
                }
                for entity in self.entities
            ],
        }

    @staticmethod
    def from_dict(config_dict):  # 修复后的方法
        return HelixConfig(
            job_name=config_dict["job_name"],
            recycle=config_dict.get("recycle", 10),
            ensemble=config_dict.get("ensemble", 1),
            entities=[
                HelixProteinConfig(
                    type=entity.type,
                    sequence=entity.sequence,
                    count=entity.count,
                    modification=entity.modification,
                )
                for entity in config_dict.get("entities", [])
            ],
        )


################################## RFAA #####################


@dataclass
class SE3Param:
    num_layers: int = 1
    num_channels: int = 32
    num_degrees: int = 2
    l0_in_features: int = 64
    l0_out_features: int = 64
    l1_in_features: int = 3
    l1_out_features: int = 2
    num_edge_features: int = 64
    n_heads: int = 4
    div: int = 4


@dataclass
class SE3Param:
    num_layers: int = 1
    num_channels: int = 32
    num_degrees: int = 2
    l0_in_features: int = 64
    l0_out_features: int = 64
    l1_in_features: int = 3
    l1_out_features: int = 2
    num_edge_features: int = 64
    n_heads: int = 4
    div: int = 4


@dataclass
class SE3RefParam(SE3Param):
    num_layers: int = 2  # 这里显式覆盖父类的 num_layers 字段


@dataclass
class LegacyModelParam:
    n_extra_block: int = 4
    n_main_block: int = 32
    n_ref_block: int = 4
    n_finetune_block: int = 0
    d_msa: int = 256
    d_msa_full: int = 64
    d_pair: int = 192
    d_templ: int = 64
    n_head_msa: int = 8
    n_head_pair: int = 6
    n_head_templ: int = 4
    d_hidden_templ: int = 64
    p_drop: float = 0.0
    use_chiral_l1: bool = True
    use_lj_l1: bool = True
    use_atom_frames: bool = True
    recycling_type: str = "all"
    use_same_chain: bool = True
    lj_lin: float = 0.75
    SE3_param: SE3Param = field(default_factory=SE3Param)
    SE3_ref_param: SE3Param = field(default_factory=lambda: SE3Param(num_layers=2))


@dataclass
class LoaderParams:
    n_templ: int = 4
    MAXLAT: int = 128
    MAXSEQ: int = 1024
    MAXCYCLE: int = 10
    BLACK_HOLE_INIT: bool = False
    seqid: float = 150.0


@dataclass
class ChemParams:
    use_phospate_frames_for_NA: bool = True
    use_cif_ordering_for_trp: bool = True


@dataclass
class DatabaseParams:
    sequencedb: str = ""
    hhdb: str = (
        "/data/home/silong/data/alphafold/rfaa/pdb100_2021Mar03/pdb100_2021Mar03"
    )
    command: str = "make_msa.sh"  # 需要使用相对的 因为 rfaa 会用 ./ 包裹
    num_cpus: int = 12
    mem: int = 64


@dataclass
class RFAAConfig:
    job_name: str = ""
    output_path: str = ""
    checkpoint_path: str = "/data/home/silong/projects/alphafold/RoseTTAFold-All-Atom/RFAA_paper_weights.pt"
    database_params: DatabaseParams = field(default_factory=DatabaseParams)
    protein_inputs: Optional[Any] = None
    na_inputs: Optional[Any] = None
    sm_inputs: Optional[Any] = None
    covale_inputs: Optional[Any] = None
    residue_replacement: Optional[Any] = None
    chem_params: ChemParams = field(default_factory=ChemParams)
    loader_params: LoaderParams = field(default_factory=LoaderParams)
    legacy_model_param: LegacyModelParam = field(default_factory=LegacyModelParam)

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> "RFAAConfig":
        """
        创建一个 RFAAConfig 实例，从字典中加载配置。

        :param config_dict: 包含配置信息的字典
        :return: RFAAConfig 实例
        """
        return RFAAConfig(
            job_name=config_dict.get("job_name", ""),
            output_path=config_dict.get("output_path", ""),
            checkpoint_path=config_dict.get(
                "checkpoint_path",
                "/data/home/silong/projects/alphafold/RoseTTAFold-All-Atom/RFAA_paper_weights.pt",
            ),
            database_params=DatabaseParams(**config_dict.get("database_params", {})),
            protein_inputs=config_dict.get("protein_inputs"),
            na_inputs=config_dict.get("na_inputs"),
            sm_inputs=config_dict.get("sm_inputs"),
            covale_inputs=config_dict.get("covale_inputs"),
            residue_replacement=config_dict.get("residue_replacement"),
            chem_params=ChemParams(**config_dict.get("chem_params", {})),
            loader_params=LoaderParams(**config_dict.get("loader_params", {})),
            legacy_model_param=LegacyModelParam(
                **{
                    **config_dict.get("legacy_model_param", {}),
                    "SE3_param": SE3Param(
                        **config_dict.get("legacy_model_param", {}).get("SE3_param", {})
                    ),
                    "SE3_ref_param": SE3RefParam(
                        **config_dict.get("legacy_model_param", {}).get(
                            "SE3_ref_param", {}
                        )
                    ),
                }
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        将 RFAAConfig 实例转换为字典。

        :return: 包含实例属性的字典
        """
        return asdict(self)

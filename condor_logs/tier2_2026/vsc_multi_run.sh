sbatch  -A lp_biomed_mdv /scratch/leuven/350/vsc35057/projects/Balance/condor_logs/tier2/vsc_single_H100.slurm "./configs/SthSth/2mod/video_flow_Late_ShuffleGradEPIB_pre.json" "./configs/SthSth/default_config_vsc_sthsth_2mod.json"

sbatch  -A lp_biomed_mdv /scratch/leuven/350/vsc35057/projects/Synergy/condor_logs/tier2_2026/vsc_single_H100.slurm "./configs/ScienceQA/synprom_lora_synibfaster.json" "./configs/ScienceQA/default_config_scienceqa_syn_tier2.json" 0 0.0001 0.01 10 8
sbatch  -A lp_biomed_mdv /scratch/leuven/350/vsc35057/projects/Synergy/condor_logs/tier2_2026/vsc_single_H100.slurm "./configs/ScienceQA/synprom_lora_synibfaster.json" "./configs/ScienceQA/default_config_scienceqa_syn_tier2.json" 0 0.0001 0.01 1 8
sbatch  -A lp_biomed_mdv /scratch/leuven/350/vsc35057/projects/Synergy/condor_logs/tier2_2026/vsc_single_H100.slurm "./configs/ScienceQA/synprom_lora_synibfaster.json" "./configs/ScienceQA/default_config_scienceqa_syn_tier2.json" 0 0.0001 0.01 0.1 8
sbatch  -A lp_biomed_mdv /scratch/leuven/350/vsc35057/projects/Synergy/condor_logs/tier2_2026/vsc_single_H100.slurm "./configs/ScienceQA/synprom_lora_synibfaster.json" "./configs/ScienceQA/default_config_scienceqa_syn_tier2.json" 0 0.0001 0.01 0.01 8
sbatch  -A lp_biomed_mdv /scratch/leuven/350/vsc35057/projects/Synergy/condor_logs/tier2_2026/vsc_single_H100.slurm "./configs/ScienceQA/synprom_lora.json" "./configs/ScienceQA/default_config_scienceqa_syn_tier2.json" 0 0.0001 0.01 None 8
sbatch  -A lp_biomed_mdv /scratch/leuven/350/vsc35057/projects/Synergy/condor_logs/tier2_2026/vsc_single_H100.slurm "./configs/ScienceQA/synprom_lora.json" "./configs/ScienceQA/default_config_scienceqa_syn_tier2.json" 0 0.0001 0.001 None 8
sbatch  -A lp_biomed_mdv /scratch/leuven/350/vsc35057/projects/Synergy/condor_logs/tier2_2026/vsc_single_H100.slurm "./configs/ScienceQA/synprom_lora.json" "./configs/ScienceQA/default_config_scienceqa_syn_tier2.json" 0 0.001 0.01 None 8
sbatch  -A lp_biomed_mdv /scratch/leuven/350/vsc35057/projects/Synergy/condor_logs/tier2_2026/vsc_single_H100.slurm "./configs/ScienceQA/synprom_lora.json" "./configs/ScienceQA/default_config_scienceqa_syn_tier2.json" 0 0.001 0.001 None 8



sbatch  -A lp_biomed_mdv ./configs/ScienceQA/cache_lora.json ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json 0 0.0001 0.01 None 8
sbatch  -A lp_biomed_mdv ./configs/ScienceQA/cache_image_lora.json ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json 0 0.0001 0.01 None 8
sbatch  -A lp_biomed_mdv ./configs/ScienceQA/cache_text_lora.json ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json 0 0.0001 0.01 None 8

sbatch  -A lp_biomed_mdv ./configs/ScienceQA/cache_synib_lora.json ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json 0 0.0001 0.01 0.1 8 None rand 0.3
sbatch  -A lp_biomed_mdv ./configs/ScienceQA/cache_synib_lora.json ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json 0 0.0001 0.01 0.1 8 None rand 0.5
sbatch  -A lp_biomed_mdv ./configs/ScienceQA/cache_synib_lora.json ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json 0 0.0001 0.01 0.1 8 None rand 0.7
sbatch  -A lp_biomed_mdv ./configs/ScienceQA/cache_synib_lora.json ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json 0 0.0001 0.01 1 8 None rand 0.3
sbatch  -A lp_biomed_mdv ./configs/ScienceQA/cache_synib_lora.json ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json 0 0.0001 0.01 1 8 None rand 0.5
sbatch  -A lp_biomed_mdv ./configs/ScienceQA/cache_synib_lora.json ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json 0 0.0001 0.01 1 8 None rand 0.7

sbatch  -A lp_biomed_mdv ./configs/ScienceQA/cache_synib_lora.json ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json 0 0.0001 0.01 0.1 8 0.1 learned
sbatch  -A lp_biomed_mdv ./configs/ScienceQA/cache_synib_lora.json ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json 0 0.0001 0.01 1.0 8 0.1 learned
sbatch  -A lp_biomed_mdv ./configs/ScienceQA/cache_synib_lora.json ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json 0 0.0001 0.01 0.1 8 0.5 learned
sbatch  -A lp_biomed_mdv ./configs/ScienceQA/cache_synib_lora.json ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json 0 0.0001 0.01 1.0 8 0.5 learned
sbatch  -A lp_biomed_mdv ./configs/ScienceQA/cache_synib_lora.json ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json 0 0.0001 0.01 0.1 8 1 learned
sbatch  -A lp_biomed_mdv ./configs/ScienceQA/cache_synib_lora.json ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json 0 0.0001 0.01 1.0 8 1 learned
sbatch  -A lp_biomed_mdv ./configs/ScienceQA/cache_synib_lora.json ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json 0 0.0001 0.01 0.1 8 5 learned
sbatch  -A lp_biomed_mdv ./configs/ScienceQA/cache_synib_lora.json ./configs/ScienceQA/default_config_scienceqa_cache_tier2.json 0 0.0001 0.01 1.0 8 5 learned



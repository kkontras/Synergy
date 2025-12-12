#!/bin/sh
#module swap cluster/dodrio/debug_rome
#module swap cluster/dodrio/gpu_rome_a100
#module swap cluster/dodrio/gpu_rome_a100_40
module swap cluster/dodrio/gpu_rome_a100_80
#module swap cluster/dodrio/debug_rome

#qsub ./condor_logs/tier1/jobfile.pbs -v config="configs/SthSth/Late/video_flow_layout_Late_ind_AGM.json",default_config="./configs/SthSth/default_config_vsctier1_sthsth.json"
#qsub ./condor_logs/tier1/jobfile_2gpu.pbs -v config="configs/SthSth/Late/video_flow_layout_Late_ind_AGM_a1.json",default_config="./configs/SthSth/default_config_vsctier1_sthsth.json"
#qsub ./condor_logs/tier1/jobfile_4gpu.pbs -v config="configs/SthSth/Late/video_flow_layout_Late_ind_AGM_a2.json",default_config="./configs/SthSth/default_config_vsctier1_sthsth.json"


#qsub ./condor_logs/tier1/debug_jobfile.pbs -v config="./configs/SthSth/paperready/video_flow_layout_Late_ShuffleGradEPIB_pre_ind.json",default_config="./configs/SthSth/default_config_vsctier1_sthsth.json",l=0.001,multil=1,l_diffsq=0,lib=0,num_samples=32,reg_by=dist_pred_3d_agree,batch_size=16,contr_coeff=1,shuffle_type=rand,contr_type=label
qsub ./condor_logs/tier1/jobfile.pbs -v config="./configs/SthSth/paperready/video_flow_layout_Late_ShuffleGradEPIB_pre_ind.json",default_config="./configs/SthSth/default_config_vsctier1_sthsth.json",l=0.001,multil=1,l_diffsq=0,lib=0,num_samples=32,reg_by=dist_pred_3d_agree,batch_size=16,contr_coeff=1,shuffle_type=rand,contr_type=label
qsub ./condor_logs/tier1/jobfile.pbs -v config="./configs/SthSth/paperready/video_flow_layout_Late_ShuffleGradEPIB_pre_ind.json",default_config="./configs/SthSth/default_config_vsctier1_sthsth.json",l=0.001,multil=1,l_diffsq=0,lib=0,num_samples=32,reg_by=dist_pred_3d_disagree,batch_size=16,contr_coeff=1,shuffle_type=rand,contr_type=label
qsub ./condor_logs/tier1/jobfile.pbs -v config="./configs/SthSth/paperready/video_flow_layout_Late_ShuffleGradEPIB_pre_ind.json",default_config="./configs/SthSth/default_config_vsctier1_sthsth.json",l=0.001,multil=1,l_diffsq=0,lib=0,num_samples=32,reg_by=dist_pred_3d_disagree_agree,batch_size=16,contr_coeff=1,shuffle_type=rand,contr_type=label


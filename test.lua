require 'torch'; require 'nn'; require 'nngraph';
require 'cudnn'; require 'cunn'; require 'image';

matio = require 'matio'

-------------------------------------------------
model_info = {}

-- ** Depending on models you want to test, modify <model_info.name and .resol> as follows.

-- model_info.name = {'resol2_vggL1', 'resol2_onlyL1', 'resol1_vggL1', 'resol1_onlyL1'}
-- model_info.resol = {2, 2, 1, 1}

-- model_info.name = {'resol2_vggL1', 'resol1_vggL1'}
-- model_info.resol = {2, 1}

model_info.name = {'resol2_vggL1'}
model_info.resol = {2}

-------------------------------------------------
path_info = {}
path_info.resultFolder = 'result_test'
path_info.model = 'models_trained'
path_info.dataFolder = 'data/unifiedDB' -- used in src_test/sub2_1_DataLoad.lua
path_info.dataFolder_test = 'data/testDB' -- used in src_test/sub2_1_DataLoad.lua

path_info.utils = 'src_train/utils_v2.lua'
t = dofile(path_info.utils)

---------------------
cmd = torch.CmdLine()

cmd:option('-gpuNum', 1, 'gpu number')
cmd:option('-seed', 1, 'random seed')
cmd:option('-threads', 1, 'number of threads')

cmd:option('-batchSize_eval', 20, 'mini-batch size for evaluation')

cmd:option('-bd_thick', 2, 'plot boundary thickness')
cmd:option('-bd_color', 0, 'plot boundary color')
-- cmd:option('-bd_color_mask', 0.5, 'plot boundary color for masks')

cmd:option('-evalDataIdx', 1, '1 for LookBook full clothes, 2 for UnifiedDB segments')
cmd:option('-taskNum', 2, '1 try on given cloth | 2 change original top style | 3 change original bottom style')

cmd:text()

opt = cmd:parse(arg)

----------------------------------------------------------------
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')
if opt.gpuNum then
  cutorch.setDevice(opt.gpuNum)
end

evalIdx = {}

if opt.evalDataIdx == 1 then
  evalIdx.idxSet_g = torch.LongTensor({103, 14, 218, 914})
  evalIdx.idxSet_s = torch.LongTensor({2003, 2103, 1153, 1094})

  evalIdx.dataFlag_g, evalIdx.dataFlag_s = 'fsSyn', 'lookBk'
elseif opt.evalDataIdx == 2 then
  evalIdx.idxSet_g = torch.LongTensor({3, 72})
  evalIdx.idxSet_s = torch.LongTensor({198, 221})

  evalIdx.dataFlag_g, evalIdx.dataFlag_s = 'fsSyn', 'dpWear'
end


evalIdx.nData = evalIdx.idxSet_g:size(1)
if evalIdx.nData ~= evalIdx.idxSet_s:size(1) then
  error(' ** numSample for g and s should be the same -> check')
end

print(' data'..opt.evalDataIdx..': # = '..evalIdx.nData..' '..evalIdx.dataFlag_s)

---------------------
path_info.save = paths.concat(path_info.resultFolder, opt.evalDataIdx..'_'..evalIdx.dataFlag_s)

if opt.taskNum == 1 then -- task1. try on given cloth | our result with vitonStage1-mask
  print(' *** task1. tryOn givenCloth')
  path_info.save = path_info.save..'_t1-vtnMask'

elseif opt.taskNum == 2 then -- task2. change top style | our result with original top-mask
  print(' *** task2. change OrigTopStyle')
  path_info.save = path_info.save..'_t2-origTop'

elseif opt.taskNum == 3 then -- task3. change bottom style | our result with original btm-mask
  print(' *** task3. change OrigBtmStyle')
  path_info.save = path_info.save..'_t3-origBtm'
end


print('- save to: '..path_info.save)

if (path.exists(path_info.resultFolder) == false) then path.mkdir(path_info.resultFolder) end
if (path.exists(path_info.save) == false) then path.mkdir(path_info.save) end


---------------------
dofile 'src_test/sub2_1_Setup_DataLoad.lua'
dofile 'src_test/sub2_2_DefineFunc.lua'

print(' ')


---------------------
opt.n_models = #model_info.name
for m_idx = 1, opt.n_models do

  local time_all = sys.clock()

  local path_model = paths.concat(path_info.model, model_info.name[m_idx],
                                    'model_minValidLoss.t7')
  local save_flag = model_info.name[m_idx]

  print('--------------------------------------------')
  print(m_idx..'/'..opt.n_models..' | load '.. save_flag)

  model = torch.load(path_model)
  if opt.gpuNum then model:cuda() end

  opt.resol = model_info.resol[m_idx]
  if opt.resol == 1 then -- 1 for 192x64
    opt.t_scale = torch.FloatTensor({197, 66}) -- [h,w]
    opt.t_crop = torch.FloatTensor({192, 64}) -- [h,w]
  elseif opt.resol == 2 then -- 2 for 384x128
    opt.t_scale = torch.FloatTensor({394, 132}) -- [h,w]
    opt.t_crop = torch.FloatTensor({384, 128}) -- [h,w]
  end

  func_scale = t.Scale_fixWfixH(opt.t_scale)
  func_scale_cropSz = t.Scale_fixWfixH(opt.t_crop)
  func_CenterCrop_img = t.CenterCrop_img_fixWfixH(opt.t_crop)
  func_CenterCrop_mask = t.CenterCrop_mask_fixWfixH(opt.t_crop)

  print('  input resol '..opt.resol..': '..opt.t_crop[1]..' '..opt.t_crop[2])

  --------------------------------------------------------------------------------------------------------------
  print('  do forward')

  local evalSet_g = evalIdx.idxSet_g
  local evalSet_s = evalIdx.idxSet_s

  local s_raw, g_raw, y_fg_raw, person_vis, y_person_vis, mCloth_origRes = func_tryOn( evalSet_g, evalSet_s )

  s_raw:add(1):div(2)
  y_fg_raw:add(1):div(2)

  local dd = mCloth_origRes[{{},{1},{},{}}]:clone()
  local s_raw_whiteBg = t.ApplyMask(dd)(s_raw:clone())
  local m_bg = func_chCopy_binary2color(func_make_m_bg(dd))
  s_raw_whiteBg = s_raw_whiteBg + m_bg

    ------------------
  path_info.save_final = paths.concat(path_info.save, save_flag)
  if (path.exists(path_info.save_final) == false) then path.mkdir(path_info.save_final) end

  func_saveFig_v1(person_vis, '0_ps', 'each', evalSet_g, evalSet_s, path_info.save_final)
  func_saveFig_v1(g_raw, '1a_gRaw', 'each', evalSet_g, evalSet_s, path_info.save_final)
  func_saveFig_v1(s_raw, '2a_sRaw', 'each', evalSet_g, evalSet_s, path_info.save_final)
  func_saveFig_v1(s_raw_whiteBg, '2b_sRaw_whiteBg', 'each', evalSet_g, evalSet_s, path_info.save_final)
  func_saveFig_v1(y_fg_raw, '3a_y', 'each', evalSet_g, evalSet_s, path_info.save_final)
  func_saveFig_v1(y_person_vis, '3b_y_ps', 'each', evalSet_g, evalSet_s, path_info.save_final)


  func_saveFig_v2_all(person_vis, g_raw, s_raw_whiteBg,
  y_fg_raw, y_person_vis, evalSet_g, evalSet_s, path_info.save_final, opt.bd_color, opt.bd_thick)

  print(string.format('    time_all  %.1f s', sys.clock() - time_all))

end
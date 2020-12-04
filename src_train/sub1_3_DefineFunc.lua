
dofile 'src_train/subsub1_3_DataAug.lua'


function func_train(epoch)

  local shuffle = torch.randperm(opt.n_train):long()

  local nBatchTotal = math.ceil(opt.n_train/opt.batchSize)
  local del_epoch = 1/nBatchTotal

  local parameters, gradParameters = model:getParameters()
  model:training()

  local time_train = sys.clock()
  local s_idx_train, e_idx_train

  for idx_batch = 1, nBatchTotal do

    if opt.flag_printBatch == 0 then
      xlua.progress(idx_batch, nBatchTotal)
    end

    s_idx_train = 1 + (idx_batch-1)*opt.batchSize
    e_idx_train = math.min(idx_batch*opt.batchSize, opt.n_train)
    local n_batch = e_idx_train - s_idx_train + 1

    --------------------------------------------------------
    local x_fg_target, x_fg_input, m_fg_input
      = func_makeInput('train', 1, shuffle, s_idx_train, e_idx_train)
    --------------------------------------------------------

    local feval = function (x)

      local err = {}
      err.L1, err.vgg = 0, 0;

      if x~=parameters then parameters:copy(x) end
      gradParameters:zero()

      local y_fg = model:forward({m_fg_input, x_fg_input})

      -------------------------------------------------------------
      err.L1 = crit_L1:forward(y_fg, x_fg_target)
      local df_do = crit_L1:backward(y_fg, x_fg_target)
      df_do = df_do:mul(opt.lamb_L1)   
      
      -------------------------------------------------------------
      
      if opt.vgg == 1 then
      
        -- ** preprocessing for VggNet: [-1, 1] -> [0, 255] & BGR & MeanZero
        local t_vgg = m_vggPrep:forward(x_fg_target):clone()
        local x_vgg = m_vggPrep:forward(y_fg)
              
        -- ** forward & backward in VggNet
        err.vgg = crit_vgg:forward(x_vgg, {content_target=t_vgg})
        local df_do_vgg =crit_vgg:backward(x_vgg, {content_target=t_vgg})
        
        df_do_vgg = df_do_vgg:mul(opt.lamb_vgg)            
        
        -- compensation of prep. scaling in grad    
        df_do_vgg = m_vggPrep:updateGradInput(y_fg, df_do_vgg)
        
        -------------------------------------------------------------            
        df_do = df_do + df_do_vgg
        
      end
      
      model:backward({m_fg_input, x_fg_input}, df_do)
      err.all = (opt.lamb_L1 * err.L1) + (opt.lamb_vgg * err.vgg)

      func_update_info_batch(epoch, err, idx_batch, del_epoch)

      return err.all, gradParameters

    end -- end for feval

    optimFunc(feval, parameters, optimState)

    ----------------
    if opt.flag_printBatch == 1 then
      print(string.format('eph %d/%d bat %d/%d %d~%d: all %.3f (L1 %.3f VGG %.3f)',
        epoch, opt.nEpoch, idx_batch, nBatchTotal,
        s_idx_train, e_idx_train,
        info_batch.err_all[-1], info_batch.err_L1[-1], info_batch.err_vgg[-1]))
    end

    x_fg_target, x_fg_input, m_fg_input = nil, nil, nil

  end -- end for idx_batch = 1, nBatchTotal do

  collectgarbage()

end -- end for func_train

--------------------------------------------------------
function func_eval(data_type)

  model:clearState()
  model:evaluate()

  local err_eval = {}
  local temp_sz

  if data_type == 1 then temp_sz = opt.n_train
  elseif data_type == 2 then temp_sz = opt.n_valid
  elseif data_type == 3 then temp_sz = opt.n_test
  end

  local idx_arr_eval = torch.range(1, temp_sz, 1)
  local e_L1, e_vgg = 0, 0;

  for ss = 1, temp_sz, opt.batchSize_eval do

    xlua.progress(ss, temp_sz)

    local s_idx_eval = ss
    local e_idx_eval = math.min(ss+opt.batchSize_eval-1, temp_sz)
    local nn = e_idx_eval-s_idx_eval+1

    --------------------------------------------------------
    local x_fg_target, x_fg_input, m_fg_input
      = func_makeInput('eval', data_type, idx_arr_eval, s_idx_eval, e_idx_eval)
    --------------------------------------------------------

    local y_fg = model:forward({m_fg_input, x_fg_input})
    
    local temp_e_L1 = crit_L1:forward(y_fg, x_fg_target)
    e_L1 = e_L1 + (temp_e_L1 * nn)
    
    -------------------------------------------------------------
    local t_vgg, x_vgg  
    if opt.vgg == 1 then  
    
      -- ** preprocessing for VggNet: [-1, 1] -> [0, 255] & BGR & MeanZero
      t_vgg = m_vggPrep:forward(x_fg_target):clone()
      x_vgg = m_vggPrep:forward(y_fg)

      -- ** forward in VggNet
      local temp_e_vgg = crit_vgg:forward(x_vgg, {content_target=t_vgg})
      e_vgg = e_vgg + (temp_e_vgg * nn)      
    
    end
    
    -------------------------------------------------------------

    x_fg_target, x_fg_input, m_fg_input = nil, nil, nil
    y_fg, t_vgg, x_vgg = nil, nil, nil

  end -- end for batch

  err_eval.L1 = e_L1 / (temp_sz)
  err_eval.vgg = e_vgg / (temp_sz)
  err_eval.all = (opt.lamb_L1 * err_eval.L1) + (opt.lamb_vgg * err_eval.vgg)

  collectgarbage()

  return err_eval

end

--------------------------------------------------------
function func_update_info(epoch, err_eval, time_eval, time_train)

  print(' ')
  print('=============================================')
  print(string.format('Eph %d/%d, timeT = %.1fs, timeE = %.1fs',
    epoch, opt.nEpoch, time_train, time_eval))
  print(string.format('  Valid all %.3f (L1 %.3f VGG %.3f)',
    err_eval.all, err_eval.L1, err_eval.vgg))


  if epoch == 0 then
    info.epoch = torch.FloatTensor({epoch})
    
    info.err_all_valid = torch.FloatTensor({err_eval.all})
    info.err_L1_valid = torch.FloatTensor({err_eval.L1})
    info.err_vgg_valid = torch.FloatTensor({err_eval.vgg})

    info.time_train = torch.FloatTensor({0})
    info.time_eval = torch.FloatTensor({time_eval})

  else
    info.epoch = torch.cat(info.epoch, torch.FloatTensor({epoch}), 1)
    
    info.err_all_valid = torch.cat(info.err_all_valid, torch.FloatTensor({err_eval.all}), 1)
    info.err_L1_valid = torch.cat(info.err_L1_valid, torch.FloatTensor({err_eval.L1}), 1)
    info.err_vgg_valid = torch.cat(info.err_vgg_valid, torch.FloatTensor({err_eval.vgg}), 1)

    info.time_train = torch.cat(info.time_train, torch.FloatTensor({time_train}), 1)
    info.time_eval = torch.cat(info.time_eval, torch.FloatTensor({time_eval}), 1)

  end

end


function func_update_info_batch(epoch, err, idx_batch, del_epoch)

  if epoch == 1 and idx_batch == 1 then
    info_batch = {}
    info_batch.epoch = torch.FloatTensor({epoch})
    info_batch.idx_batch = torch.FloatTensor({idx_batch})
    
    info_batch.err_all = torch.FloatTensor({err.all})
    info_batch.err_L1 = torch.FloatTensor({err.L1}) -- before weighting
    info_batch.err_vgg = torch.FloatTensor({err.vgg}) -- before weighting 

    info_batch.epoch_batchCount = torch.FloatTensor({ del_epoch*idx_batch + (epoch-1) })
  else
    info_batch.epoch = torch.cat(info_batch.epoch, torch.FloatTensor({epoch}), 1)
    info_batch.idx_batch = torch.cat(info_batch.idx_batch, torch.FloatTensor({idx_batch}), 1)
    
    info_batch.err_all = torch.cat(info_batch.err_all, torch.FloatTensor({err.all}), 1)
    info_batch.err_L1 = torch.cat(info_batch.err_L1, torch.FloatTensor({err.L1}), 1) -- before weighting    
    info_batch.err_vgg = torch.cat(info_batch.err_vgg, torch.FloatTensor({err.vgg}), 1) -- before weighting   

    info_batch.epoch_batchCount = torch.cat(info_batch.epoch_batchCount,
      torch.FloatTensor({ del_epoch*idx_batch + (epoch-1) }), 1)
  end

end



function func_save_onlyMinVal(epoch)
  if epoch == 1 or ( info.err_all_valid[{epoch+1}] < (info.err_all_valid[{{2, epoch}}]:min()) ) then
    print('eval standard: err_all_valid (NOT recon L1)')
    print(string.format('=> Save @ eph %d: minValidLoss', epoch)); print(' ')

    torch.save(paths.concat(path_info.save, 'model_minValidLoss.t7'), model)
    matio.save(paths.concat(path_info.save, 'infoEpoch_minValidLoss.mat'), info)
    matio.save(paths.concat(path_info.save, 'infoBatch_minValidLoss.mat'), info_batch)
    torch.save(paths.concat(path_info.save, 'infoEpoch_minValidLoss.t7'), info)
    torch.save(paths.concat(path_info.save, 'infoBatch_minValidLoss.t7'), info_batch)
  else
    print('eval standard: err_all_valid (NOT recon L1)')
    print(string.format('=> Try minValEval @ eph %d: BUT NotMin -> NoSave', epoch)); print(' ')    
  end
end

function func_save_epoch(epoch)

  print(string.format('=> Save @ eph %d', epoch))
  torch.save(paths.concat(path_info.save, 'model_eph'..epoch..'.t7'), model)
  matio.save(paths.concat(path_info.save, 'infoEpoch_eph'..epoch..'.mat'), info)
  torch.save(paths.concat(path_info.save, 'infoEpoch_eph'..epoch..'.t7'), info)

  if epoch > 0 then
    matio.save(paths.concat(path_info.save, 'infoBatch_eph'..epoch..'.mat'), info_batch)
    torch.save(paths.concat(path_info.save, 'infoBatch_eph'..epoch..'.t7'), info_batch)
  end

  if epoch == opt.nEpoch then
    info.time_all = sys.clock() - info.time_all
    print(string.format('     ------ TimeAll %.1fs', info.time_all))

    if opt.continueFlag == 0 or info.time_all_arr == nil then
      info.time_all_arr = torch.FloatTensor({info.time_all})
    else
      info.time_all_arr = torch.cat(info.time_all_arr, torch.FloatTensor({info.time_all}), 1)
    end

    torch.save(paths.concat(path_info.save, 'final_model_eph'..epoch..'.t7'), model)
    matio.save(paths.concat(path_info.save, 'final_infoEpoch_eph'..epoch..'.mat'), info)
    matio.save(paths.concat(path_info.save, 'final_infoBatch_eph'..epoch..'.mat'), info_batch)
    torch.save(paths.concat(path_info.save, 'final_infoEpoch_eph'..epoch..'.t7'), info)
    torch.save(paths.concat(path_info.save, 'final_infoBatch_eph'..epoch..'.t7'), info_batch)
  end

  if epoch > 1 then
  
    local del_epoch = epoch - 1
    local temp_path = paths.concat(path_info.save, 'model_eph'..del_epoch..'.t7')
  
    while(paths.filep(temp_path) == false) do
      del_epoch = del_epoch - 1
      temp_path = paths.concat(path_info.save, 'model_eph'..del_epoch..'.t7')
    end
    print(string.format('    *** delEpoch = %d', del_epoch))
  
    os.remove(paths.concat(path_info.save, 'infoEpoch_eph'..(del_epoch)..'.mat'))
    os.remove(paths.concat(path_info.save, 'infoBatch_eph'..(del_epoch)..'.mat'))
    os.remove(paths.concat(path_info.save, 'infoEpoch_eph'..(del_epoch)..'.t7'))
    os.remove(paths.concat(path_info.save, 'infoBatch_eph'..(del_epoch)..'.t7'))
    os.remove(paths.concat(path_info.save, 'model_eph'..(del_epoch)..'.t7'))
  end

end

--------------------------------------------------------

function func_plot_learnCurve(epoch)
  gnuplot.pngfigure(paths.concat(path_info.save_plot, 'LearnCurve.png'))
  gnuplot.title(string.format('Learning Curve @ Epoch%d',epoch))
  gnuplot.plot(
    {'Train (Batch)',  info_batch.epoch_batchCount,  info_batch.err_all,  '-'},
    {'Valid', info.epoch, info.err_all_valid, '-'})
  gnuplot.xlabel('Epoch')
  gnuplot.ylabel('Loss')
  gnuplot.plotflush()
end


function func_data_exImg(data_type)

  local X = {}

  local data_str
  if data_type == 1 then data_str = 'train'
  elseif data_type == 2 then data_str = 'valid'
  elseif data_type == 3 then data_str = 'test'
    local imgSet_g_1_TopInner = torch.LongTensor({82, 881})
    local imgSet_g_2_TopOuter = torch.LongTensor({889, 62})
    local imgSet_g_3_Bottom = torch.LongTensor({154, 158})
    local imgSet_g_4_WholeInner = torch.LongTensor({10, 2})
    local imgSet_g_5_WholeOuter = torch.LongTensor({123, 24})
    X.imgSet_c = torch.LongTensor({846, 91, 83, 85, 92, 37})
    X.imgSet_g = torch.cat({imgSet_g_1_TopInner, imgSet_g_2_TopOuter, imgSet_g_3_Bottom,
      imgSet_g_4_WholeInner, imgSet_g_5_WholeOuter, X.imgSet_c}, 1)
  end

  local s_idx_eval, e_idx_eval = 1, X.imgSet_g:size(1)
  local g_x_fg_target, g_x_fg_input, g_m_fg_input
    = func_makeInput('eval', data_type, X.imgSet_g, s_idx_eval, e_idx_eval)

  local s_idx_eval, e_idx_eval = 1, X.imgSet_c:size(1)
  local c_x_fg_target, c_x_fg_input, c_m_fg_input
    = func_makeInput('eval', data_type, X.imgSet_c, s_idx_eval, e_idx_eval)

  X.g_m_fg_input = g_m_fg_input:clone()
  X.c_x_fg_target = c_x_fg_target:clone()
  X.c_x_fg_input = c_x_fg_input:clone()

  g_x_fg_target, g_x_fg_input, g_m_fg_input = nil, nil, nil
  c_x_fg_target, c_x_fg_input, c_m_fg_input = nil, nil, nil

  return X

end



function func_batch2img(X, catDir)
  -- X: Nx3xHxW | X[idx]: 3xHxW
  -- -- catDir 2: height dir. | catDir 3: width dir.
  if X:dim() == 3 then
    return X
  elseif X:dim() == 4 then
    local X_cat
    local n_sample = X:size(1)
    for idx  = 1, n_sample do
      if idx == 1 then X_cat = X[idx]
      else X_cat = torch.cat(X_cat, X[idx], catDir)
      end
    end
    return X_cat
  end
end


function func_plot_exImg(data_type, epoch, X)
  model:clearState()
  model:evaluate()

  local data_str
  if data_type == 1 then data_str = 'train'
  elseif data_type == 2 then data_str = 'valid'
  elseif data_type == 3 then data_str = 'test'
  end
  print('  --- plot reconEx: eph '..epoch..' '..data_str)

  local plot_result

  local num_g = X.imgSet_g:size(1)
  local num_c = X.imgSet_c:size(1)

  for idx_x_c = 1, num_c, 1 do -- for every content input

    local temp_x_c = X.c_x_fg_input[idx_x_c]:clone()
    local x_c_copy = temp_x_c:repeatTensor(num_g, 1, 1, 1)

    local y_fg = model:forward({X.g_m_fg_input, x_c_copy}) -- min:-1, max:+1, Nx3xHxW

    --------------------------------------------------------------
    local y_fg_reScale = y_fg:clone():float():add(1):div(2) -- min:0, max:+1
    y_fg_reScale = func_clipToRange_r1r2(y_fg_reScale, 1, 0) -- for safe
    local y_fg_reScale_cat = func_batch2img(y_fg_reScale, 3)

    --------------------------------------------------------------

    local temp_t_c_VisNoNorm = X.c_x_fg_target[idx_x_c]:clone():float():add(1):div(2)
    temp_t_c_VisNoNorm = func_clipToRange_r1r2(temp_t_c_VisNoNorm, 1, 0) -- for safe
    local temp_x_c_VisNoNorm = temp_x_c:clone():float():add(1):div(2)
    temp_x_c_VisNoNorm = func_clipToRange_r1r2(temp_x_c_VisNoNorm, 1, 0) -- for safe

    local temp_plot_result = torch.cat({temp_t_c_VisNoNorm, temp_x_c_VisNoNorm, y_fg_reScale_cat}, 3)

    if idx_x_c == 1 then
      plot_result = temp_plot_result
    else
      plot_result = torch.cat(plot_result, temp_plot_result, 2)
    end

    temp_x_c, x_c_copy, y_fg, y_fg_reScale, y_fg_reScale_cat = nil, nil, nil, nil, nil
    temp_t_c_VisNoNorm, temp_x_c_VisNoNorm, temp_plot_result = nil, nil, nil

  end -- for idx_x_c = 1, num_c, 1 do -- for every content input


  local temp_path = paths.concat(path_info.save_plot,
    string.format('ex_'..data_str..'_x_g.jpg'))

  if (paths.filep(temp_path) == false) then

    local temp_g_m = func_batch2img(X.g_m_fg_input, 3)
    temp_g_m = temp_g_m:float():add(1):div(2)
    image.save(temp_path, temp_g_m)

    temp_g_m = nil
  end

  local temp_path = paths.concat(path_info.save_plot,
    string.format('ex_'..data_str..'_eph%d.jpg', epoch))
  image.save(temp_path, plot_result)

  plot_result = nil

end

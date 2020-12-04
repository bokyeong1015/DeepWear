
opt.t_scale_viton = torch.FloatTensor({256, 192}) -- [h,w]
func_scale_viton = t.Scale_fixWfixH(opt.t_scale_viton)

--------------------------------------------------------
function func_chCopy_binary2color(x)
  if x:dim() == 3 then -- 1xHxW
    x = torch.cat({x,x,x}, 1)
  elseif x:dim() == 4 then -- Nx1xHxW
    x = torch.cat({x,x,x}, 2)
  end
  return x
end

function func_addBottomBd_v1(X, thick_bottom, thick_color)       
  local sz = X:size() -- 3xHxW
  local temp_boundary = torch.FloatTensor(3,  thick_bottom, sz[3]):fill(thick_color)
  
  X = torch.cat({X, temp_boundary}, 2) -- vertical cat    
  return X 
end 

function func_batch2img_addRightBd_v1(X, bd_color, bd_thick, flag_reSz)
  -- X: Nx3xHxW | X[idx]: 3xHxW
  -- -- catDir 2: height dir. | catDir 3: width dir.

  local temp_boundary, temp_h
  if flag_reSz == 1 then temp_h = opt.t_scale_viton[1]
  else temp_h = opt.t_crop[1]
  end
  temp_boundary = torch.FloatTensor(3,  temp_h, bd_thick):fill(bd_color):typeAs(X)

  if X:dim() == 3 then    
    local X_temp = X:clone()
    if flag_reSz == 1 then X_temp = func_scale_viton(X_temp) end      
    
    X_temp = torch.cat({X_temp, temp_boundary}, 3) -- horizontal cat
    return X_temp

  elseif X:dim() == 4 then    
    local X_cat
    local n_sample = X:size(1)
    for idx  = 1, n_sample do

      local X_temp = X[idx]:clone()      
      if flag_reSz == 1 then X_temp = func_scale_viton(X_temp) end   
         
      if idx ~= n_sample then X_temp = torch.cat({X_temp, temp_boundary}, 3) end -- horizontal cat

      if idx == 1 then X_cat = X_temp
      else X_cat = torch.cat(X_cat, X_temp, 3)
      end

    end
    return X_cat

  end

end

function func_bbox(input_mask)

  local n_sample = input_mask:size(1)
  local bbox_scale = torch.FloatTensor(n_sample, 4)
  
  for kk = 1, n_sample do
    local m_fg_idx = torch.nonzero(input_mask[kk]:eq(1))

    local h1 = torch.min(m_fg_idx[{{},{2}}])
    local h2 = torch.max(m_fg_idx[{{},{2}}])
    local w1 = torch.min(m_fg_idx[{{},{3}}])
    local w2 = torch.max(m_fg_idx[{{},{3}}])
    
    bbox_scale[kk] = torch.FloatTensor({h1, h2, w1, w2})
            
  end
  
  return bbox_scale

end


--------------------------------------------------------
local flag_resize = 1

function func_saveFig_v1(x, data_saveFlag, saveOpt, evalSet_g, evalSet_s, save_path)
  if saveOpt == 'each' then
    local nData = x:size(1)
    for kk = 1, nData do
      local g_idx = evalSet_g[kk]
      local s_idx = evalSet_s[kk]
      local x_temp = x[kk]

      local temp_path

      if data_saveFlag == '0_ps' or data_saveFlag == '1a_gRaw' then
          temp_path = paths.concat(save_path, 'g'..g_idx..'_'..data_saveFlag..'.jpg')
      elseif data_saveFlag == '2b_sRaw_whiteBg' or data_saveFlag == '2a_sRaw' then
         temp_path = paths.concat(save_path, 's'..s_idx..'_'..data_saveFlag..'.jpg')
      else
         temp_path = paths.concat(save_path, 'g'..g_idx..'s'..s_idx..'_'..data_saveFlag..'.jpg')
      end

      if flag_resize == 1 then x_temp = func_scale_viton(x_temp) end

      image.save(temp_path, x_temp)
    end

  end

end


function func_saveFig_v2_all(p_vis, x_g, x_s, y_fg, y_p, evalSet_g, evalSet_s, save_path, bd_color, bd_thick)
  local temp_h
  if flag_resize == 1 then temp_h = opt.t_scale_viton[1]
  else temp_h = opt.t_crop[1]
  end

  local bd = torch.FloatTensor(3,  temp_h, bd_thick):fill(bd_color):typeAs(p_vis)

  local nData = p_vis:size(1)
  for kk = 1, nData do
      local g_idx = evalSet_g[kk]
      local s_idx = evalSet_s[kk]

      local a1, a2, a3, a4, a5 = p_vis[kk], x_g[kk], x_s[kk], y_fg[kk], y_p[kk]
      if flag_resize == 1 then
        a1, a2, a3, a4, a5 = func_scale_viton(a1), func_scale_viton(a2), func_scale_viton(a3), func_scale_viton(a4), func_scale_viton(a5)
      end
      local x_temp = torch.cat({a1, bd, a2, bd, a3, bd, a4, bd, a5}, 3) -- horizontal cat

      local temp_path = paths.concat(save_path, 'all_g'..g_idx..'s'..s_idx..'.jpg')
      image.save(temp_path, x_temp)
  end

end

--------------------------------------------------------
function func_tryOn(idxSet_g, idxSet_s)

  model:clearState()
  model:evaluate()
    
  local temp_sz = idxSet_g:size(1)
   
  local x_fg_input, m_cloth
  local m_orig_top, m_orig_btm, m_viton, x_person_vis
  local bbox_re, y_fg, x_fg_target
  local dummy_m_vtn, dummy_out_vtn, m_fg_input
  local dpWr_x_fg_target, dpWr_bbox_s
  
  local final_raw_y_fg = torch.FloatTensor(temp_sz, 3, opt.t_crop[1], opt.t_crop[2])
  local final_raw_s = torch.FloatTensor(temp_sz, 3, opt.t_crop[1], opt.t_crop[2])
  local final_raw_m = torch.FloatTensor(temp_sz, 3, opt.t_crop[1], opt.t_crop[2])
  local final_raw_person_vis = torch.FloatTensor(temp_sz, 3, opt.t_crop[1], opt.t_crop[2])
  local final_raw_y_person = torch.FloatTensor(temp_sz, 3, opt.t_crop[1], opt.t_crop[2])
  
  local final_mCloth_origRes = torch.FloatTensor(temp_sz, 3, opt.t_crop[1], opt.t_crop[2])


  for ss = 1, temp_sz, opt.batchSize_eval do

    xlua.progress(ss, temp_sz)

    local s_idx_eval = ss
    local e_idx_eval = math.min(ss+opt.batchSize_eval-1, temp_sz)
    local nn = e_idx_eval-s_idx_eval+1
    
    local eval_arr_g = idxSet_g[{{s_idx_eval, e_idx_eval}}]
    local eval_arr_s = idxSet_s[{{s_idx_eval, e_idx_eval}}]
    
    
    -----------------------------------------------------
    if evalIdx.dataFlag_s == 'lookBk' then -- lookbook cloth for style input
      x_fg_input, m_cloth = func_lookBk_evalData(eval_arr_s)

    elseif evalIdx.dataFlag_s == 'dpWear' then -- unifiedDB segment for style input
      x_fg_input, dpWr_x_fg_target = func_dpWear_evalData_style(3, eval_arr_s, 1)
      m_cloth, dpWr_bbox_s = func_dpWear_evalData_geom(3, eval_arr_s, 1)
    end
    
    final_mCloth_origRes[{{s_idx_eval, e_idx_eval},{},{},{}}] = func_chCopy_binary2color(m_cloth:clone()):float()
    ----------------------------------------------------

    m_orig_top, m_orig_btm, m_viton, x_person_vis = func_fsSyn_evalData(eval_arr_g, eval_arr_s)    

    if opt.taskNum == 1 then -- task1. try on given cloth | our result with vitonStage1-mask
    
        m_fg_input = m_viton
        bbox_re = func_bbox(m_viton)
        y_fg = model:forward({m_viton, x_fg_input}) 

    elseif opt.taskNum == 2 then -- task2. change top style | our result with original top-mask
    
        m_fg_input = m_orig_top
        bbox_re = func_bbox(m_orig_top)            
        y_fg = model:forward({m_orig_top, x_fg_input}) 
        
    elseif opt.taskNum == 3 then -- task3. change bottom style | our result with original btm-mask
        
        m_fg_input = m_orig_btm
        bbox_re = func_bbox(m_orig_btm)            
        y_fg = model:forward({m_orig_btm, x_fg_input})

    end
    
    final_raw_y_fg[{{s_idx_eval, e_idx_eval},{},{},{}}] = y_fg:clone():float()
    final_raw_s[{{s_idx_eval, e_idx_eval},{},{},{}}] = x_fg_input:clone():float()
    final_raw_m[{{s_idx_eval, e_idx_eval},{},{},{}}] = func_chCopy_binary2color(m_fg_input):float() 
    final_raw_person_vis[{{s_idx_eval, e_idx_eval},{},{},{}}] = x_person_vis:float()      
    
    local m_bg = func_make_m_bg(m_fg_input)
    local x_bg = t.ApplyMask(m_bg)(x_person_vis:clone())
    local y_fg_plot = y_fg:clone()
    y_fg_plot = y_fg_plot:add(1):div(2)
    y_fg_plot = t.ApplyMask(m_fg_input)(y_fg_plot)
    y_fg_plot = y_fg_plot + x_bg
    final_raw_y_person[{{s_idx_eval, e_idx_eval},{},{},{}}] = y_fg_plot:float()

    -----------------------------------------------------------------

  end
    
  return final_raw_s, final_raw_m, final_raw_y_fg, final_raw_person_vis, final_raw_y_person, final_mCloth_origRes
end
    
func_scale = t.Scale_fixWfixH(opt.t_scale)
func_LocRandCrop = t.out_loc_RandomCrop_fixWfixH(opt.t_scale, opt.t_crop)
func_CenterCrop_img = t.CenterCrop_img_fixWfixH(opt.t_crop)
func_CenterCrop_mask = t.CenterCrop_mask_fixWfixH(opt.t_crop)

------------------------------------------------------------------
function func_loadEachSample(data_type, ext_idx)

  local temp_x_raw, temp_m_fg, h1, h2, w1, w2

  if data_type == 1 then
    temp_x_raw = image.load(path_img.raw.train[ext_idx], 3, 'float') -- 3xHxW, min 0, max 1
    temp_m_fg = torch.load(path_img.mask_fg.train[ext_idx]) -- ByteTensor - size: HxW
    
    h1 = path_img.bbox.train[{ext_idx,1}]
    h2 = path_img.bbox.train[{ext_idx,2}]
    w1 = path_img.bbox.train[{ext_idx,3}]
    w2 = path_img.bbox.train[{ext_idx,4}]

  elseif data_type == 2 then
    temp_x_raw = image.load(path_img.raw.valid[ext_idx], 3, 'float')
    temp_m_fg = torch.load(path_img.mask_fg.valid[ext_idx])
    
    h1 = path_img.bbox.valid[{ext_idx,1}]
    h2 = path_img.bbox.valid[{ext_idx,2}]
    w1 = path_img.bbox.valid[{ext_idx,3}]
    w2 = path_img.bbox.valid[{ext_idx,4}]

  elseif data_type == 3 then
    temp_x_raw = image.load(path_img.raw.test[ext_idx], 3, 'float')
    temp_m_fg = torch.load(path_img.mask_fg.test[ext_idx])
    
    h1 = path_img.bbox.test[{ext_idx,1}]
    h2 = path_img.bbox.test[{ext_idx,2}]
    w1 = path_img.bbox.test[{ext_idx,3}]
    w2 = path_img.bbox.test[{ext_idx,4}]

  end

  return temp_x_raw, temp_m_fg, h1, h2, w1, w2
  

end


function func_makeInput(flag_phase, data_type, idx_arr, s_idx, e_idx)

  local n_sample = e_idx - s_idx + 1

  local x_fg_target = torch.FloatTensor(n_sample, 3, opt.t_crop[1], opt.t_crop[2])
  local x_fg_input = torch.FloatTensor(n_sample, 3, opt.t_crop[1], opt.t_crop[2])
  local m_fg_input = torch.FloatTensor(n_sample, 1, opt.t_crop[1], opt.t_crop[2])

  local temp_s = idx_arr[{{s_idx, e_idx}}]

  for kk = 1, temp_s:size(1) do

    local temp_x_raw, temp_m_fg, h1, h2, w1, w2
      = func_loadEachSample(data_type, temp_s[kk]) -- temp_x_raw 3xHxW (e.g., 3x810x192) | min 0, max 1

    -- content input
    temp_x_raw = temp_x_raw:clone():mul(2):add(-1) -- min -1, max 1
    local temp_x_fg = t.ApplyMask(temp_m_fg)(temp_x_raw) -- 3xHxW, min -1, max 1
    local temp_x_fg_input = temp_x_fg[{{},{h1,h2},{w1,w2}}]
    temp_x_fg_input = func_scale(temp_x_fg_input)
    temp_x_fg_input = func_clipToRange_r1r2(temp_x_fg_input, 1, -1) -- for safe

    -- geometry input
    local temp_m_fg_input = func_scale(temp_m_fg) -- min 0, max 1
    temp_m_fg_input = temp_m_fg_input:round()

    -- computing target image
    local orig_sz = temp_x_raw:size() -- 3xHxW
    temp_x_raw = func_scale(temp_x_raw) -- min -1, max 1

    local temp_x_fg_target = t.ApplyMask(temp_m_fg_input)(temp_x_raw)
    temp_x_fg_target = func_clipToRange_r1r2(temp_x_fg_target, 1, -1) -- for safe

    if flag_phase == 'train' then

      if opt.aug_train == 1 then
          temp_m_fg_input, temp_x_fg_target = func_augVert(temp_m_fg_input, temp_x_fg_target,
            orig_sz, h1, h2, w1, w2, 1)

          -- temp_m_fg_input, temp_x_fg_target = func_augPosScale(temp_m_fg_input, temp_x_fg_target,
          --    orig_sz, h1, h2, w1, w2, 1)
      end

      temp_x_fg_target, temp_x_fg_input, temp_m_fg_input
      = func_augFlipCrop(opt.aug_train, temp_x_fg_target, temp_x_fg_input, temp_m_fg_input)
      -- 0 for no aug (centCrop), 1 for horFlip+randCrop

    elseif flag_phase == 'eval' then
      temp_x_fg_target, temp_x_fg_input, temp_m_fg_input
      = func_augFlipCrop(0, temp_x_fg_target, temp_x_fg_input, temp_m_fg_input)
    end

    x_fg_target[kk] = temp_x_fg_target
    x_fg_input[kk] =  temp_x_fg_input
    m_fg_input[kk] = temp_m_fg_input

    temp_x_raw, temp_x_fg_target = nil, nil
    temp_x_fg, temp_x_fg_input = nil, nil
    temp_m_fg, temp_m_fg_input = nil, nil

    h1, h2, w1, w2 = nil, nil, nil, nil

  end

  if opt.gpuNum then
    x_fg_target = x_fg_target:cuda()
    x_fg_input = x_fg_input:cuda()
    m_fg_input = m_fg_input:cuda()
  end

  collectgarbage()

  return x_fg_target, x_fg_input, m_fg_input
end



function func_augFlipCrop(aug_flag, temp_x_fg_target, temp_x_fg_input, temp_m_fg_input)

  local inputRand_hFlip = torch.uniform() -- M.HorFlip_inpRand(prob, inputRand)
  local out_bg = func_LocRandCrop()
  local out_fg = func_LocRandCrop()

  if aug_flag == 0 then -- no augmentation, center crop only
    temp_x_fg_target = func_CenterCrop_img(temp_x_fg_target) 
    temp_x_fg_input = func_CenterCrop_img(temp_x_fg_input) 
    temp_m_fg_input = func_CenterCrop_mask(temp_m_fg_input) 

  elseif aug_flag == 1 then -- horizontal flip + translation (random crop)
    -- the same horFlip for x_c, x_g, y

    temp_x_fg_target = t.HorFlip_inpRand(0.5, inputRand_hFlip)(temp_x_fg_target)
    temp_x_fg_input = t.HorFlip_inpRand(0.5, inputRand_hFlip)(temp_x_fg_input)
    temp_m_fg_input = t.HorFlip_inpRand(0.5, inputRand_hFlip)(temp_m_fg_input)

    temp_x_fg_target = t.RandCrop_wLoc(out_bg)(temp_x_fg_target) 
    temp_x_fg_input = t.RandCrop_wLoc(out_fg)(temp_x_fg_input) 
    temp_m_fg_input = t.RandCrop_wLoc(out_bg)(temp_m_fg_input) 

  end

  return temp_x_fg_target, temp_x_fg_input, temp_m_fg_input

end


------------------------------------------------------------------
function func_augVert(temp_m_fg_input, temp_x_fg_target, orig_sz, h1, h2, w1, w2, aug_prob)

  if torch.uniform() < aug_prob then
    -- opt.t_scale = torch.FloatTensor({197, 66}) -- [h,w]
    local h1_new = torch.round(h1*opt.t_scale[1]/orig_sz[2]);
    local h2_new = torch.round(h2*opt.t_scale[1]/orig_sz[2]);
    local w1_new = torch.round(w1*opt.t_scale[2]/orig_sz[3]);
    local w2_new = torch.round(w2*opt.t_scale[2]/orig_sz[3]);

    if h1_new < 1 then h1_new = 1; end
    if h2_new > opt.t_scale[1] then h2_new = opt.t_scale[1]; end
    if w1_new < 1 then w1_new = 1; end
    if w2_new > opt.t_scale[2] then w2_new = opt.t_scale[2]; end

    local m_seg = temp_m_fg_input[{{h1_new,h2_new},{w1_new,w2_new}}]:clone()
    local y_seg = temp_x_fg_target[{{},{h1_new,h2_new},{w1_new,w2_new}}]:clone()

    ----
    local del_h = h2_new - h1_new + 1;
    local del_w = w2_new - w1_new + 1;

    local w1_newnew = torch.random(1, opt.t_scale[2] - del_w+1);
    local h1_newnew = torch.random(1, opt.t_scale[1] - del_h+1);
    local w2_newnew = w1_newnew + del_w - 1;
    local h2_newnew = h1_newnew + del_h - 1;

    local mm = torch.FloatTensor(opt.t_scale[1], opt.t_scale[2]):fill(0)
    local yy = torch.FloatTensor(3, opt.t_scale[1], opt.t_scale[2]):fill(0)

    mm[{{h1_newnew,h2_newnew},{w1_newnew,w2_newnew}}] = m_seg
    yy[{{},{h1_newnew,h2_newnew},{w1_newnew,w2_newnew}}] = y_seg

    temp_m_fg_input = mm:clone()
    temp_x_fg_target = yy:clone()

    mm, m_seg, yy, y_seg = nil, nil, nil, nil

  end

  return temp_m_fg_input, temp_x_fg_target

end


function func_augPosScale(temp_m_fg_input, temp_x_fg_target, orig_sz, h1, h2, w1, w2, aug_prob)

  if torch.uniform() < aug_prob then
    local h1_new = torch.round(h1*opt.t_scale[1]/orig_sz[2]);
    local h2_new = torch.round(h2*opt.t_scale[1]/orig_sz[2]);
    local w1_new = torch.round(w1*opt.t_scale[2]/orig_sz[3]);
    local w2_new = torch.round(w2*opt.t_scale[2]/orig_sz[3]);

    if h1_new < 1 then h1_new = 1; end
    if h2_new > opt.t_scale[1] then h2_new = opt.t_scale[1]; end
    if w1_new < 1 then w1_new = 1; end
    if w2_new > opt.t_scale[2] then w2_new = opt.t_scale[2]; end

    local m_seg = temp_m_fg_input[{{h1_new,h2_new},{w1_new,w2_new}}]:clone()
    local y_seg = temp_x_fg_target[{{},{h1_new,h2_new},{w1_new,w2_new}}]:clone()
    
    ----    
    local del_h = h2_new - h1_new + 1; 
    local del_w = w2_new - w1_new + 1; 

    local scale_h = torch.random(torch.round(del_h/2), opt.t_scale[1])
    local scale_w = torch.random(torch.round(del_w/2), opt.t_scale[2])  
    
    m_seg = image.scale(m_seg, scale_w, scale_h, 'bicubic')-- w, h
    y_seg = image.scale(y_seg, scale_w, scale_h, 'bicubic')-- w, h

    
    ----
    local w1_newnew = torch.random(1, opt.t_scale[2] - scale_w+1);
    local h1_newnew = torch.random(1, opt.t_scale[1] - scale_h+1);
    local w2_newnew = w1_newnew + scale_w - 1;
    local h2_newnew = h1_newnew + scale_h - 1;

    local mm = torch.FloatTensor(opt.t_scale[1], opt.t_scale[2]):fill(0)
    local yy = torch.FloatTensor(3, opt.t_scale[1], opt.t_scale[2]):fill(0)

    mm[{{h1_newnew,h2_newnew},{w1_newnew,w2_newnew}}] = m_seg
    yy[{{},{h1_newnew,h2_newnew},{w1_newnew,w2_newnew}}] = y_seg

    temp_m_fg_input = mm:clone()
    temp_x_fg_target = yy:clone()

    mm, m_seg, yy, y_seg = nil, nil, nil, nil
    
  end
  
  return temp_m_fg_input, temp_x_fg_target
    
end 


------------------------------------------------------------------
function func_clipToRange01(x)
  -- x: a single image or mask
  local y = x:clone();
  y = torch.reshape(y, torch.numel(x))
  y[y:gt(1)] = 1 -- greater than
  y[y:lt(0)] = 0 -- lower than
  y = torch.reshape(y, x:size())

  x = nil

  return y
end

------------------------------------------------------------------
function func_clipToRange_r1r2(x, r1, r2)
  -- x: a single image or mask
  -- to ensure r2 <= x <= r1
  local y = x:clone();
  y = torch.reshape(y, torch.numel(x))
  y[y:gt(r1)] = r1
  y[y:lt(r2)] = r2
  y = torch.reshape(y, x:size())

  x = nil

  return y
end

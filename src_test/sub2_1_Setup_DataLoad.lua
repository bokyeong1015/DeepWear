
data_info_fsSyn = {}

data_info_fsSyn.person = paths.concat(path_info.dataFolder_test, 'fsSyn_img')
data_info_fsSyn.reMask = paths.concat(path_info.dataFolder_test, 'fsSyn_mask')

opt.data_fsSyn = {}
opt.data_fsSyn.n_data = 2374

path_img_fsSyn = {}
path_img_fsSyn.person_img = {};

for idx = 1, opt.data_fsSyn.n_data do

  local name_personImg = string.format('%d_p.jpg', idx)
  local path_personImg = paths.concat(data_info_fsSyn.person, name_personImg)

  table.insert(path_img_fsSyn.person_img, path_personImg)

end

-------------------------------------------------------------------
if opt.evalDataIdx == 1 then
    data_info_lookBk = {}
    data_info_lookBk.cloth =paths.concat(path_info.dataFolder_test, 'lkbk_img_mask')

    opt.data_lookBk = {}
    opt.data_lookBk.n_data = 2374

    path_img_lookBk = {}
    path_img_lookBk.cloth_img = {};
    path_img_lookBk.cloth_mask = {};

    for idx = 1, opt.data_lookBk.n_data do

      local name_clothImg = string.format('%d_c.jpg', idx)
      local name_clothMask = string.format('%d_m.png', idx)
      local path_clothImg = paths.concat(data_info_lookBk.cloth, name_clothImg)
      local path_clothMask = paths.concat(data_info_lookBk.cloth, name_clothMask)

      table.insert(path_img_lookBk.cloth_img, path_clothImg)
      table.insert(path_img_lookBk.cloth_mask, path_clothMask)

    end
end
-------------------------------------------------------------------
if opt.evalDataIdx == 2 then
    path_info.opt = paths.concat(path_info.dataFolder, 'data_opt.mat')
    opt_db = matio.load(path_info.opt, 'opt')

    path_info.bbox = paths.concat(path_info.dataFolder, 'data_bbox.mat')
    local bbox_db = matio.load(path_info.bbox, 'bbox_mat')

    path_info.dataFolder_train = paths.concat(path_info.dataFolder, 'train');
    path_info.dataFolder_valid = paths.concat(path_info.dataFolder, 'valid');
    path_info.dataFolder_test = paths.concat(path_info.dataFolder, 'test');

    opt.n_train = opt_db.n_train[{1,1}]
    opt.n_valid = opt_db.n_valid[{1,1}]
    opt.n_test = opt_db.n_test[{1,1}]


    path_img = {}
    path_img.raw = {}; path_img.bbox = {}; path_img.mask_fg = {};

    path_img.raw.train = {}; path_img.mask_fg.train = {}; path_img.bbox.train = bbox_db.train
    path_img.raw.valid = {}; path_img.mask_fg.valid = {}; path_img.bbox.valid = bbox_db.valid
    path_img.raw.test = {}; path_img.mask_fg.test = {}; path_img.bbox.test = bbox_db.test

    for idx = 1, opt.n_train do
      local file_name_raw = string.format('%d_raw.jpg', idx)
      local file_name_fg = string.format('%d_mask_fg.t7', idx)

      local file_path_raw = paths.concat(path_info.dataFolder_train, file_name_raw)
      local file_path_fg = paths.concat(path_info.dataFolder_train, file_name_fg)
      table.insert(path_img.raw.train, file_path_raw)
      table.insert(path_img.mask_fg.train, file_path_fg)
    end

    for idx = 1, opt.n_valid do
      local file_name_raw = string.format('%d_raw.jpg', idx)
      local file_name_fg = string.format('%d_mask_fg.t7', idx)

      local file_path_raw = paths.concat(path_info.dataFolder_valid, file_name_raw)
      local file_path_fg = paths.concat(path_info.dataFolder_valid, file_name_fg)
      table.insert(path_img.raw.valid, file_path_raw)
      table.insert(path_img.mask_fg.valid, file_path_fg)
    end

    for idx = 1, opt.n_test do
      local file_name_raw = string.format('%d_raw.jpg', idx)
      local file_name_fg = string.format('%d_mask_fg.t7', idx)

      local file_path_raw = paths.concat(path_info.dataFolder_test, file_name_raw)
      local file_path_fg = paths.concat(path_info.dataFolder_test, file_name_fg)
      table.insert(path_img.raw.test, file_path_raw)
      table.insert(path_img.mask_fg.test, file_path_fg)
    end
end

-------------------------------------------------------------------
function func_fsSyn_loadEach_person(ext_idx)
  local x_person = image.load(path_img_fsSyn.person_img[ext_idx]) 
  return x_person
end


function func_fsSyn_loadEach_origMask(ext_idx)
  local p_idx = ext_idx
  local path_m_top = paths.concat(data_info_fsSyn.reMask, 'p'..p_idx..'_m-orig_1-top.png');
  local path_m_btm = paths.concat(data_info_fsSyn.reMask, 'p'..p_idx..'_m-orig_2-btm.png');
  local m_orig_top = image.load(path_m_top)
  local m_orig_btm = image.load(path_m_btm)
  
  return m_orig_top, m_orig_btm
end 

function func_fsSyn_loadEach_vitonMask(ext_idx_p, ext_idx_c)
  local p_idx = ext_idx_p  
  local c_idx = ext_idx_c
  
  local data_str
  if evalIdx.dataFlag_s == 'lookBk' then
    data_str = 'lookbook'
  elseif evalIdx.dataFlag_s == 'dpWear' then
    data_str = 'deepWear'
  end
  
  local path_m = paths.concat(data_info_fsSyn.reMask, 'p'..p_idx..'_m-'..data_str..'_c'..c_idx..'.png');
  local m_viton = image.load(path_m)

  return m_viton
end 



function func_fsSyn_evalData(eval_arr_person, eval_arr_cloth)
  
  local n_sample = eval_arr_person:size(1)
  
  local x_person_vis = torch.FloatTensor(n_sample, 3, opt.t_crop[1], opt.t_crop[2]):fill(0)
  local m_orig_top = torch.FloatTensor(n_sample, 1, opt.t_crop[1], opt.t_crop[2])
  local m_orig_btm = torch.FloatTensor(n_sample, 1, opt.t_crop[1], opt.t_crop[2])
  local m_viton = torch.FloatTensor(n_sample, 1, opt.t_crop[1], opt.t_crop[2])

  for kk = 1, n_sample do
  
    local temp_x_person = func_fsSyn_loadEach_person(eval_arr_person[kk])    

    local temp_m_orig_top, temp_m_orig_btm = func_fsSyn_loadEach_origMask(eval_arr_person[kk])    
    local temp_m_viton = func_fsSyn_loadEach_vitonMask(eval_arr_person[kk], eval_arr_cloth[kk])    

    temp_m_orig_top = func_scale_cropSz(temp_m_orig_top)
    m_orig_top[kk] = temp_m_orig_top:round()

    temp_m_orig_btm = func_scale_cropSz(temp_m_orig_btm)
    m_orig_btm[kk] = temp_m_orig_btm:round()
    
    temp_m_viton = func_scale_cropSz(temp_m_viton)
    m_viton[kk] = temp_m_viton:round()

    temp_x_person = func_scale_cropSz(temp_x_person)
    x_person_vis[kk] = func_clipToRange_r1r2(temp_x_person, 1, 0)
    
  end
   
  if opt.gpuNum then  
    m_orig_top = m_orig_top:cuda()  
    m_orig_btm = m_orig_btm:cuda()    
    m_viton = m_viton:cuda()
    x_person_vis = x_person_vis:cuda()
  end

  return m_orig_top, m_orig_btm, m_viton, x_person_vis
end



function func_lookBk_loadEach(ext_idx)
  local x_cloth = image.load(path_img_lookBk.cloth_img[ext_idx])
  local m_cloth = image.load(path_img_lookBk.cloth_mask[ext_idx])
  
  return x_cloth, m_cloth
end

function func_lookBk_evalData(eval_arr)

  local n_sample = eval_arr:size(1)
  
  local x_cloth = torch.FloatTensor(n_sample, 3, opt.t_crop[1], opt.t_crop[2])
  local m_cloth = torch.FloatTensor(n_sample, 1, opt.t_crop[1], opt.t_crop[2]):fill(0)  

  for kk = 1, n_sample do

    local temp_x_cloth, temp_m_cloth = func_lookBk_loadEach(eval_arr[kk])

    temp_x_cloth = temp_x_cloth:mul(2):add(-1)
    temp_x_cloth = t.ApplyMask(temp_m_cloth)(temp_x_cloth)
    temp_x_cloth = func_scale_cropSz(temp_x_cloth)
    x_cloth[kk] = func_clipToRange_r1r2(temp_x_cloth, 1, -1)


    temp_m_cloth = func_scale_cropSz(temp_m_cloth)
    m_cloth[kk] = temp_m_cloth:round()
    
  end

  if opt.gpuNum then
    x_cloth = x_cloth:cuda()
    m_cloth = m_cloth:cuda()    
  end

  return x_cloth, m_cloth
end


-------------------------------------------------------------------
function func_dpWear_loadEach_geom(data_type, ext_idx)

  local temp_m_fg, bbox

  if data_type == 1 then
    temp_m_fg = torch.load(path_img.mask_fg.train[ext_idx])
    bbox = path_img.bbox.train[{{ext_idx}, {}}]

  elseif data_type == 2 then
    temp_m_fg = torch.load(path_img.mask_fg.valid[ext_idx])
    bbox = path_img.bbox.valid[{{ext_idx}, {}}]

  elseif data_type == 3 then
    temp_m_fg = torch.load(path_img.mask_fg.test[ext_idx])
    bbox = path_img.bbox.test[{{ext_idx}, {}}]
  end

  return temp_m_fg, bbox
end

function func_dpWear_loadEach_style(data_type, ext_idx)

  local temp_x_raw, temp_m_fg, h1, h2, w1, w2

  if data_type == 1 then
    temp_x_raw = image.load(path_img.raw.train[ext_idx], 3, 'float')
    temp_m_fg = torch.load(path_img.mask_fg.train[ext_idx])

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


function func_dpWear_evalData_style(data_type, eval_arr, flag_tgtImg)

  local n_sample = eval_arr:size(1)
  
  local x_fg_input = torch.FloatTensor(n_sample, 3, opt.t_crop[1], opt.t_crop[2])
  local x_fg_target = torch.FloatTensor(n_sample, 3, opt.t_crop[1], opt.t_crop[2]):fill(0)
  

  for kk = 1, n_sample do

    local temp_x_raw, temp_m_fg, h1, h2, w1, w2 = func_dpWear_loadEach_style(data_type, eval_arr[kk])

    temp_x_raw = temp_x_raw:clone():mul(2):add(-1)
    local temp_x_fg = t.ApplyMask(temp_m_fg)(temp_x_raw)
    local temp_x_fg_input = temp_x_fg[{{},{h1,h2},{w1,w2}}]
    temp_x_fg_input = func_scale(temp_x_fg_input)
    temp_x_fg_input = func_clipToRange_r1r2(temp_x_fg_input, 1, -1)
    
    x_fg_input[kk] =  func_CenterCrop_img(temp_x_fg_input)
    
    if flag_tgtImg == 1 then
      local temp_m_fg_input = func_scale(temp_m_fg)
      temp_m_fg_input = temp_m_fg_input:round()

      local orig_sz = temp_x_raw:size()
      temp_x_raw = func_scale(temp_x_raw)
  
      local temp_x_fg_target = t.ApplyMask(temp_m_fg_input)(temp_x_raw)
      temp_x_fg_target = func_clipToRange_r1r2(temp_x_fg_target, 1, -1)
      
      x_fg_target[kk] = func_CenterCrop_img(temp_x_fg_target)
    end
    
  end

  if opt.gpuNum then
    x_fg_input = x_fg_input:cuda()
    x_fg_target = x_fg_target:cuda()    
  end

  return x_fg_input, x_fg_target
end

function func_dpWear_evalData_geom(data_type, eval_arr, flag_bbox)

  local n_sample = eval_arr:size(1)
  local m_fg_input = torch.FloatTensor(n_sample, 1, opt.t_crop[1], opt.t_crop[2])
  local bbox_scale = torch.FloatTensor(n_sample, 4)
  
  local h_del = math.ceil((opt.t_scale[1] - opt.t_crop[1])/2)
  local w_del = math.ceil((opt.t_scale[2] - opt.t_crop[2])/2)
  

  for kk = 1, n_sample do

    local temp_m_fg, bbox = func_dpWear_loadEach_geom(data_type, eval_arr[kk])

    local temp_m_fg_input = func_scale(temp_m_fg)
    temp_m_fg_input = temp_m_fg_input:round()

    m_fg_input[kk] = func_CenterCrop_mask(temp_m_fg_input)

    if flag_bbox == 1 then      
      local orig_sz = temp_m_fg:size()
      local h1, h2, w1, w2 = bbox[{1,1}], bbox[{1,2}], bbox[{1,3}], bbox[{1,4}]
      
      local h1_new = torch.round(h1*opt.t_scale[1]/orig_sz[1]) - h_del + 1;
      local h2_new = torch.round(h2*opt.t_scale[1]/orig_sz[1]) - h_del;
      local w1_new = torch.round(w1*opt.t_scale[2]/orig_sz[2]) - w_del + 1;
      local w2_new = torch.round(w2*opt.t_scale[2]/orig_sz[2]) - w_del;
      
      if h1_new < 1 then h1_new = 1; end
      if h2_new > opt.t_crop[1] then h2_new = opt.t_crop[1]; end
      if w1_new < 1 then w1_new = 1; end
      if w2_new > opt.t_crop[2] then w2_new = opt.t_crop[2]; end

      
      local del_pixel = 4
      if w1_new == w2_new or w1_new > w2_new then
        w2_new = w1_new + del_pixel
      end
      if h1_new == h2_new or h1_new > h2_new then
        h2_new = h1_new + del_pixel
      end

      bbox_scale[kk] = torch.FloatTensor({h1_new, h2_new, w1_new, w2_new})

      
    end

  end

  if opt.gpuNum then
    m_fg_input = m_fg_input:cuda()
  end

  return m_fg_input, bbox_scale
end


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

function func_make_m_bg(m_fg)

  local m_bg = m_fg:clone()
  
  m_bg:mul(2):add(-1)
  m_bg[m_bg:eq(1)]=0
  m_bg[m_bg:eq(-1)]=1
  
  return m_bg
end
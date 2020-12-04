
local M = {}

------------------------------------------------------------------------------

-- Parse a string of comma-separated numbers
-- For example convert "1.0,3.14" to {1.0, 3.14}
function M.parse_num_list(s)
  local nums = {}
  for _, ss in ipairs(s:split(',')) do
    table.insert(nums, tonumber(ss))
  end
  return nums
end


-- Parse a layer string and associated weights string.
-- The layers string is a string of comma-separated layer strings, and the
-- weight string contains comma-separated numbers. If the weights string
-- contains only a single number it is duplicated to be the same length as the
-- layers.
function M.parse_layers(layers_string, weights_string)
  local layers = layers_string:split(',')
  local weights = M.parse_num_list(weights_string)
  if #weights == 1 and #layers > 1 then
    -- Duplicate the same weight for all layers
    local w = weights[1]
    weights = {}
    for i = 1, #layers do
      table.insert(weights, w)
    end
  elseif #weights ~= #layers then
    local msg = 'size mismatch between layers "%s" and weights "%s"'
    error(string.format(msg, layers_string, weights_string))
  end
  return layers, weights
end

------------------------------------------------------------------------------

function M.HorFlip_inpRand(prob, inputRand)
  return function(input)
    if inputRand < prob then
      input = image.hflip(input)
    end
    return input
  end
end

function M.out_loc_RandomCrop_fixWfixH(sizeOrig, sizeCrop)
  return function()
    local w, h = sizeOrig[2], sizeOrig[1]
    local x1, y1 = torch.random(0, w - sizeCrop[2]), torch.random(0, h - sizeCrop[1])
    local out = {}
    out.x1 = x1
    out.y1 = y1
    out.x2 = x1 + sizeCrop[2]
    out.y2 = y1 + sizeCrop[1]
    return out
  end
end

function M.Scale_fixWfixH(size, interpolation)
  interpolation = interpolation or 'bicubic'
  return function(input)
    local input_sc = input:clone()
    return image.scale(input_sc, size[2], size[1], interpolation) -- w, h
  end
end

function M.RandCrop_wLoc(out)
  return function(input)
    input = input:clone()
    local x1 = out.x1
    local y1 = out.y1
    local x2 = out.x2
    local y2 = out.y2
    return image.crop(input, x1, y1, x2, y2)
  end
end

-- Crop to centered rectangle
function M.CenterCrop_img_fixWfixH(sizeCrop)
  return function(input)
    input = input:clone()
    local w1 = math.ceil((input:size(3) - sizeCrop[2])/2)
    local h1 = math.ceil((input:size(2) - sizeCrop[1])/2)
    return image.crop(input, w1, h1, w1 + sizeCrop[2], h1 + sizeCrop[1]) -- center patch
  end
end

-- Crop to centered rectangle
function M.CenterCrop_mask_fixWfixH(sizeCrop)
  return function(input)
    input = input:clone()
    local w1 = math.ceil((input:size(2) - sizeCrop[2])/2)
    local h1 = math.ceil((input:size(1) - sizeCrop[1])/2)
    return image.crop(input, w1, h1, w1 + sizeCrop[2], h1 + sizeCrop[1]) -- center patch
  end
end


function M.ApplyMask(mask)
  return function(img)
    if img:dim() ~= 4 then
      img = img:clone()
      for i=1,3 do
        img[i]:cmul(mask)
      end
      return img
    else
      img = img:clone()
      for kk = 1, img:size(1) do
        local temp_img = img[kk]:clone()
        local temp_mask = mask[kk]:clone()
        for i = 1,3 do
          temp_img[i]:cmul(temp_mask)
        end
        img[kk] = temp_img:clone()
        temp_img, temp_mask = nil, nil
      end
      return img
    end
  end
end


return M

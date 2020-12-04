
local VGGpreprocess, parent = torch.class('nn.VGGpreprocess', 'nn.Module')

-- Preprocess an image before passing to a VGG model.
-- refer to 'fast_neural_style.preprocess'

function VGGpreprocess:__init()
  parent.__init(self)
  self.vgg_mean = {103.939, 116.779, 123.68} -- BGR mean
  self.perm = torch.LongTensor{3, 2, 1} -- for RGB to BGR
end


function VGGpreprocess:updateOutput(input)
  
  -- y = {((x+1)/2) * 255} - m where x is in [-1, 1] & m is mean 
  -- input: NxCxHxW, scale [-1, 1], RGB ch order 
  
  local mean = input.new(self.vgg_mean):view(1, 3, 1, 1):expandAs(input)
  
  self.output:resizeAs(input)  
  self.output:copy(input)
  
  self.output:index(2, self.perm) -- RGB to BGR  
  
  self.output:add(1):mul(127.5):add(-1, mean)

  return self.output
end


function VGGpreprocess:updateGradInput(input, gradOutput)
  
  -- gradOutput (= dL/dy) is given & want to compute dL/dx
  -- dL/dx = (dL/dy) * (dy/dx) & (dy/dx) = 255/2 = 127.5
  
  self.gradInput:resizeAs(gradOutput)
  self.gradInput:copy(gradOutput)
  
  self.gradInput:mul(127.5)
  self.gradInput:index(2, self.perm) -- BGR to RGB
  
  return self.gradInput
end



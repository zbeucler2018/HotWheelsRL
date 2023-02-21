
-- By: Zack Beucler

totalFrames = 0



function isDone()
	-- taking too long	
	if totalFrames >= 10000 then
		return true
	end

	local single_lap = 330
	local three_laps = 950
	local LAP_LIMIT = single_lap
	if data.progress >= LAP_LIMIT then
		return true
	else
		return false
	end
end


previous_progress = 0
function calculateReward()
	local current_progress = data.progress
	if current_progress > previous_progress then
		local delta = current_progress - previous_progress
		previous_progress = current_progress
		return delta
	else
		return 0
	end
	totalFrames = totalFrames + 1
end





function dino_multi_isDone()
	local gameover_val = 33583680
	if data.score > gameover_val then
		return true
	else
		return false
	end
end

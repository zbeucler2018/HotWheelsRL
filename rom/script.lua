
-- By: Zack Beucler

function isDone()
	-- local gameover_val = 33583680
	-- if data.score > gameover_val then
	-- 	return true
	-- else
	-- 	return false
	-- end

	local single_lap = 320
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
		return delta * 10
	else
		return 0
	end
end



-- By: Zack Beucler
-- single lap: (data.progress == 320)
-- three laps: (data.progress >= 950)


function isGameOver()
	-- if data.progress >= 949 then
	if data.lap >= 4 then
		return true
	else
		return false
	end
end


function isHittingWall()
	if data.hit_wall > 100 then
		return true
	else
		return false
	end
end


function isDone()
	return isGameOver() or isHittingWall()
end


previous_progress = 0
function calculateProgressReward()
	local current_progress = data.progress
	local delta = 0
	if current_progress > previous_progress then
		delta = current_progress - previous_progress
		previous_progress = current_progress
	end
	return delta
end


function calculateReward()
	return calculateProgressReward()
end

-- By: Zack Beucler
-- single lap: (data.progress == 320) (trex_valley)
-- three laps: (data.progress >= 950)

function isGameOver()
	if data.progress < 0 then
		--data.progress = 342 -- 1 lap on Dino Boneyard
	-- if data.lap >= 4 then
		return true
	else
		return false
	end
end


function isDone()
	prev_score = data.score
	if data.score < 0 then
		data.score = prev_score
	end
	return isGameOver()
end


previous_progress = 0
function calculateProgressReward()
	local current_progress = data.progress
	local delta = 0
	if current_progress > previous_progress then
		delta = 1 -- current_progress - previous_progress
		previous_progress = current_progress
	end
	return delta
end


function calculateReward()
	return calculateProgressReward()
end

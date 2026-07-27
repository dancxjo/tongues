import test from "node:test";import assert from "node:assert/strict";
import {committedTurnAction,latencySnapshot} from "./live-conversation-model.mjs";
const event=(type,text,role="recognition")=>({type,data:{role,text}});
test("unstable ASR never becomes a user turn",()=>assert.equal(committedTurnAction({event:event("partial_hypothesis","hel"),playbackActive:false}).action,"wait"));
test("committed external speech starts or barges in",()=>{
  assert.equal(committedTurnAction({event:event("committed_segment","hello"),playbackActive:false,bargeIn:true}).action,"start_turn");
  assert.equal(committedTurnAction({event:event("committed_segment","stop please"),playbackActive:true,bargeIn:true,spokenText:"Speech starts now"}).action,"cancel_and_restart");
});
test("target-like committed echo does not interrupt",()=>assert.equal(committedTurnAction({event:event("committed_segment","speech starts now"),playbackActive:true,bargeIn:true,spokenText:"Speech starts now"}).action,"ignore_echo"));
test("latency categories remain independent",()=>assert.deepEqual(latencySnapshot({captureStarted:0,speechStarted:10,firstPartial:20,committed:40,firstGenerated:50,firstPlanned:60,firstAudio:80,playbackCompleted:120}),{
  auditory_detection_ms:10,segmentation_ms:30,asr_first_partial_ms:10,llm_first_token_ms:10,speech_planning_ms:10,tts_first_audio_ms:20,playback_ms:40,
}));

'use client'

import { Text, Slider, Button} from '@mantine/core'


function AudioControl({volume, setVolume, speaking, setSpeaking}: {volume: number, setVolume: (v: number) => void, speaking: boolean, setSpeaking: (s: boolean) => void}) {

  return (
    <>
      <Text>音量</Text>
      <Slider
      label="Volume"
      value={volume}
      onChange={setVolume}
      min={0}
      max={1}
      step={0.01}
      thumbSize={14}
    ></Slider>
    <Button onClick={() => {setSpeaking(!speaking)}}>{speaking ? '停止する' : '再生する'}</Button>
</>)
}

export default AudioControl

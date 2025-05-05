'use client'

import { Text, Slider, Button} from '@mantine/core'


function AudioControl({volume, setVolume, speaking, setSpeaking}: {volume: number, setVolume: (v: number) => void, speaking: boolean, setSpeaking: (s: boolean) => void}) {

  return (
    <>
      <Text>Volume</Text>
      <Slider
      label="Volume"
      value={volume}
      onChange={setVolume}
      min={0}
      max={1}
      step={0.01}
      thumbSize={14}
    ></Slider>
    <Button onClick={() => {setSpeaking(!speaking)}}>{speaking ? 'Stop' : 'Start'}</Button>
</>)
}

export default AudioControl

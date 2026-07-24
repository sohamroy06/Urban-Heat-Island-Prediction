import { useEffect, useState } from 'react';
import { useMotionValue, useSpring } from 'framer-motion';

export default function AnimatedNumber({ value, decimals = 1, prefix = '', suffix = '', className }) {
    const mv = useMotionValue(value);
    const spring = useSpring(mv, { stiffness: 130, damping: 22, mass: 0.6 });
    const [display, setDisplay] = useState(Number(value).toFixed(decimals));

    useEffect(() => {
        mv.set(value);
    }, [value, mv]);

    useEffect(() => {
        const unsub = spring.on('change', (v) => setDisplay(v.toFixed(decimals)));
        return unsub;
    }, [spring, decimals]);

    return (
        <span className={className}>
            {prefix}{display}{suffix}
        </span>
    );
}

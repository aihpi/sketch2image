// frontend/src/components/Icon.tsx
import React from 'react';

interface IconProps {
  name: string;
  size?: number;
  className?: string;
}

const Icon: React.FC<IconProps> = ({ name, size = 16, className = '' }) => {
  return (
    <img
      src={`/icons/${name}.svg`}
      alt=""
      width={size}
      height={size}
      className={`icon ${className}`}
      style={{ 
        display: 'block',
        flexShrink: 0 
      }}
    />
  );
};

export default Icon;
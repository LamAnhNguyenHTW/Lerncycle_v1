import type {Locale} from '@/lib/locale';
import type {Dictionary} from './types';
import {de} from './de';
import {en} from './en';

const dictionaries: Record<Locale, Dictionary> = {de, en};

export function getDictionary(locale: Locale): Dictionary {
  return dictionaries[locale];
}

export type {Dictionary} from './types';
